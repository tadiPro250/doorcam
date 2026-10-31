"""All-day hourly recording + person event clips, on their own thread so encoding never stalls the display."""
import collections
import datetime
import logging
import os
import queue
import re
import tempfile
import threading
import time

import cv2
import numpy as np

from events import EventLog

log = logging.getLogger(__name__)

NOT_NAMES = ("Unknown", "Person")
MAX_PAD_S = 5  # after a stall, repeat at most this much of the last frame


def _gst_pipeline(encoder, path, fps, kbps):
    if encoder == "nvv4l2h264enc":
        # Jetson hardware encoder; nvvidconv does the colour conversion on the GPU side.
        enc = ("videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
               "nvv4l2h264enc control-rate=1 bitrate=%d iframeinterval=%d insert-sps-pps=true"
               % (kbps * 1000, fps * 2))
    else:
        enc = ("videoconvert ! video/x-raw,format=I420 ! "
               "x264enc speed-preset=ultrafast tune=zerolatency bitrate=%d key-int-max=%d" % (kbps, fps * 2))
    # Matroska stays playable up to the last second written if the power is cut mid-file (MP4 doesn't).
    return ("appsrc ! video/x-raw,format=BGR ! queue ! %s ! h264parse ! matroskamux ! filesink location=%s"
            % (enc, path))


def open_writer(encoder, path, fps, kbps, size):
    """Returns (cv2.VideoWriter or None, actual path)."""
    if encoder == "opencv":
        path = os.path.splitext(path)[0] + ".avi"
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), float(fps), size)
    else:
        writer = cv2.VideoWriter(_gst_pipeline(encoder, path, fps, kbps), cv2.CAP_GSTREAMER, 0,
                                 float(fps), size, True)
    if not writer.isOpened():
        writer.release()
        return None, path
    return writer, path


def pick_encoder(choice, size):
    if choice != "auto":
        return choice
    if re.search(r"GStreamer:\s+YES", cv2.getBuildInformation()):
        for enc in ("nvv4l2h264enc", "x264enc"):
            probe = os.path.join(tempfile.gettempdir(), "doorcam_probe.mkv")
            writer, _ = open_writer(enc, probe, 10, 500, size)
            if writer is not None:
                writer.write(np.zeros((size[1], size[0], 3), np.uint8))
                writer.release()
                os.remove(probe)
                return enc
    log.warning("no H.264 encoder available, falling back to large MJPG .avi files")
    return "opencv"


class PacedWriter(object):
    """Keeps video time equal to real time by repeating/skipping frames to hit a fixed fps."""

    def __init__(self, writer, path, fps, frames, mono):
        """`frames` covers the time just before `mono` (the pre-roll, or only the current frame)."""
        self.writer, self.path, self.fps = writer, path, fps
        for f in frames:
            writer.write(f)
        self.slots = len(frames)
        self.t0 = mono - self.slots / float(fps)

    def feed(self, frame, mono):
        target = int((mono - self.t0) * self.fps + 1e-6)
        due = min(target - self.slots, MAX_PAD_S * self.fps)
        for _ in range(due):
            self.writer.write(frame)
        self.slots = max(self.slots, target)

    def close(self):
        self.writer.release()


def stamp(img, wall):
    text = wall.strftime("%Y-%m-%d %a %H:%M:%S")
    cv2.putText(img, text, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _safe(name):
    return re.sub(r"[^A-Za-z0-9-]+", "-", name).strip("-") or "x"


class Recorder(threading.Thread):
    def __init__(self, cfg, storage, size):
        super().__init__(name="recorder", daemon=True)
        self.cont_cfg = cfg["recording"]["continuous"]
        self.ev_cfg = cfg["recording"]["events"]
        self.cleanup_every = cfg["storage"]["cleanup_every_s"]
        self.storage = storage
        self.size = size
        self.encoder = pick_encoder(cfg["recording"]["encoder"], size)
        log.info("video encoder: %s", self.encoder)
        self.event_active = False

        self._q = queue.Queue(maxsize=30)
        self._dropped = 0
        self._halt = threading.Event()
        self._db = None
        self._last_cleanup = -1e9

        self._cont = None
        self._cont_key = None
        self._cont_retry = 0.0

        ev_fps = self.ev_cfg["fps"]
        self._buf = collections.deque(maxlen=int(self.ev_cfg["pre_roll_s"] * ev_fps))
        self._last_buf = -1e9
        self._ev = None
        self._ev_retry = 0.0
        self._last_major = -1e9

    def push(self, mono, frame, annotated, major, names):
        try:
            self._q.put_nowait((mono, frame, annotated, major, names))
        except queue.Full:
            self._dropped += 1
            if self._dropped % 100 == 1:
                log.warning("recorder is behind, dropped %d frames so far", self._dropped)

    def stop(self):
        self._halt.set()
        self.join(timeout=15)

    def run(self):
        try:
            while not self._halt.is_set():
                try:
                    item = self._q.get(timeout=1.0)
                except queue.Empty:
                    item = None
                if time.monotonic() - self._last_cleanup > self.cleanup_every:
                    self._cleanup()
                if item is None:
                    continue
                if not self.storage.available():
                    self._close_all()
                    continue
                mono, frame, annotated, major, names = item
                wall = datetime.datetime.now()
                day = self.storage.day_name(wall)
                if self.cont_cfg["enabled"]:
                    self._continuous(mono, wall, day, frame)
                if self.ev_cfg["enabled"]:
                    self._events(mono, wall, day, annotated, major, names)
        except Exception:
            log.exception("recorder crashed")
            raise
        finally:
            self._close_all()

    # -- all-day recording --------------------------------------------------

    def _continuous(self, mono, wall, day, frame):
        key = (day, wall.hour)
        if self._cont is not None and key != self._cont_key:
            self._close_cont()
        if self._cont is None:
            if mono < self._cont_retry:
                return
            path = os.path.join(self.storage.day_dir("continuous", day), wall.strftime("%H-%M-%S") + ".mkv")
            writer, path = open_writer(self.encoder, path, self.cont_cfg["fps"],
                                       self.cont_cfg["bitrate_kbps"], self.size)
            if writer is None:
                log.error("could not start recording %s, retrying in 30 s", path)
                self._cont_retry = mono + 30
                return
            self._cont = PacedWriter(writer, path, self.cont_cfg["fps"], [], mono)
            self._cont_key = key
            log.info("recording %s", path)
        if self.cont_cfg["timestamp"]:
            frame = frame.copy()
            stamp(frame, wall)
        self._cont.feed(frame, mono)

    def _close_cont(self):
        if self._cont is not None:
            self._cont.close()
            self._cont = None

    # -- person event clips -------------------------------------------------

    def _events(self, mono, wall, day, annotated, major, names):
        fps = self.ev_cfg["fps"]
        if major:
            self._last_major = mono
        if mono - self._last_buf >= 1.0 / fps:
            self._buf.append(annotated)
            self._last_buf = mono

        if self._ev is None:
            if major and mono >= self._ev_retry:
                self._start_event(mono, wall, day, annotated, names)
            return

        self._ev.feed(annotated, mono)
        if major:
            known = [n for n in names if n not in NOT_NAMES]
            if known and not self._ev_snap_known:
                cv2.imwrite(self._ev_snap, annotated)  # better snapshot: someone recognised
                self._ev_snap_known = True
            self._ev_names.update(names)
        if (mono - self._last_major > self.ev_cfg["post_roll_s"]
                or mono - self._ev_start > self.ev_cfg["max_clip_s"]):
            self._finish_event(wall)

    def _start_event(self, mono, wall, day, annotated, names):
        folder = self.storage.day_dir("events", day)
        base = os.path.join(folder, wall.strftime("%H-%M-%S"))
        writer, path = open_writer(self.encoder, base + "_recording.mkv", self.ev_cfg["fps"],
                                   self.ev_cfg["bitrate_kbps"], self.size)
        if writer is None:
            log.error("could not start event clip %s, retrying in 30 s", path)
            self._ev_retry = mono + 30
            return
        self._ev = PacedWriter(writer, path, self.ev_cfg["fps"], list(self._buf) or [annotated], mono)
        self._ev_start = mono
        self._ev_wall = wall
        self._ev_base = base
        self._ev_names = collections.Counter(names)
        self._ev_snap = base + "_recording.jpg"
        self._ev_snap_known = any(n not in NOT_NAMES for n in names)
        cv2.imwrite(self._ev_snap, annotated)
        self.event_active = True
        log.info("person event started: %s", path)

    def _finish_event(self, wall):
        ev = self._ev
        self._ev = None
        self.event_active = False
        ev.close()
        seen = set(self._ev_names)
        names = sorted(n for n in seen if n not in NOT_NAMES)
        if "Unknown" in seen:
            names.append("Unknown")
        names = names or ["Person"]
        label = "_".join(_safe(n) for n in names[:4])
        clip = "%s_%s%s" % (self._ev_base, label, os.path.splitext(ev.path)[1])
        snap = "%s_%s.jpg" % (self._ev_base, label)
        try:
            os.rename(ev.path, clip)
            os.rename(self._ev_snap, snap)
        except OSError as e:
            log.warning("could not rename event files: %s", e)
            clip, snap = ev.path, self._ev_snap
        log.info("person event saved: %s (%s)", clip, ", ".join(names))
        try:
            self._event_log().add(self._ev_wall, wall, names, clip, snap)
        except Exception as e:
            log.warning("could not write event log: %s", e)
            self._db = None

    # -- housekeeping -------------------------------------------------------

    def _event_log(self):
        if self._db is None:
            self._db = EventLog(os.path.join(self.storage.root, "events.db"))
        return self._db

    def _cleanup(self):
        self._last_cleanup = time.monotonic()
        protect = []
        if self._cont is not None:
            protect.append(self._cont.path)
        if self._ev is not None:
            protect += [self._ev.path, self._ev_snap]
        try:
            self.storage.enforce(protect)
            if self.storage.available():
                self._event_log().prune_missing()
        except Exception:
            log.exception("storage cleanup failed")
            self._db = None

    def _close_all(self):
        if self._ev is not None:
            self._finish_event(datetime.datetime.now())
        self._close_cont()
