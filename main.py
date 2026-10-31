#!/usr/bin/env python3
"""DoorCam v2: live door feed with person detection, face recognition and recording.

Run by run_doorcam.sh at login. Press Q or ESC on the monitor to quit.
"""
import argparse
import datetime
import logging
import logging.handlers
import os
import re
import signal
import sys
import threading
import time

import cv2
import yaml

import faces
import overlay
from camera import Camera
from detector import PersonDetector
from recorder import Recorder
from storage import Storage
from timesync import TimeSync
from tracker import Tracker

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WINDOW = "DoorCam"
log = logging.getLogger("doorcam")


def app_path(p):
    return p if os.path.isabs(p) else os.path.join(APP_DIR, p)


def setup_logging(cfg):
    folder = app_path(cfg["dir"])
    os.makedirs(folder, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(folder, "doorcam.log"), maxBytes=5 * 1024 * 1024, backupCount=3)
    stream_handler = logging.StreamHandler()
    root = logging.getLogger()
    root.setLevel(cfg["level"])
    for h in (file_handler, stream_handler):
        h.setFormatter(fmt)
        root.addHandler(h)


class Display(object):
    """Fullscreen window. Falls back to headless (recording only) if there is no screen."""

    def __init__(self, cfg):
        self.enabled = cfg["enabled"] and self._wait_for_x(cfg["wait_for_display_s"])
        if self.enabled:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            if cfg["fullscreen"]:
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    @staticmethod
    def _wait_for_x(timeout):
        # GTK kills the whole process if the display isn't up yet, so check first.
        disp = os.environ.get("DISPLAY")
        if not disp:
            log.warning("DISPLAY not set, running without a screen")
            return False
        m = re.match(r"^:(\d+)", disp)
        if not m:
            return True  # remote/forwarded display, can't check the socket
        sock = "/tmp/.X11-unix/X" + m.group(1)
        deadline = time.monotonic() + timeout
        while not os.path.exists(sock):
            if time.monotonic() > deadline:
                log.warning("display %s never came up, running without a screen", disp)
                return False
            time.sleep(1)
        return True

    def show(self, img):
        """Returns False when the user presses Q or ESC."""
        if not self.enabled:
            return True
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27)

    def close(self):
        if self.enabled:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.path.join(APP_DIR, "config.yaml"))
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.no_display:
        cfg["display"]["enabled"] = False
    setup_logging(cfg["logging"])
    log.info("DoorCam starting")

    halt = threading.Event()
    signal.signal(signal.SIGTERM, lambda *a: halt.set())

    timesync = TimeSync(cfg["time"])
    storage = Storage(cfg["storage"], timesync)
    size = (cfg["camera"]["width"], cfg["camera"]["height"])

    camera = Camera(cfg["camera"])
    camera.start()
    display = Display(cfg["display"])
    detector = PersonDetector(cfg["detection"])
    tracker = Tracker(cfg["tracking"])
    recorder = Recorder(cfg, storage, size)
    recorder.start()

    fcfg = cfg["faces"]
    face_worker = None
    if fcfg["enabled"]:
        if faces.available():
            face_worker = faces.FaceWorker(fcfg, app_path(fcfg["known_dir"]), storage)
            face_worker.start()
        else:
            log.warning("face_recognition is not installed, face recognition disabled")

    dcfg = cfg["detection"]
    seq = 0
    fps = 0.0
    last = time.monotonic()
    exit_code = 0
    try:
        while not halt.is_set():
            got = camera.read(seq, timeout=1.0)
            if got is None:
                if not display.show(overlay.message_frame(size, "Waiting for camera...")):
                    break
                continue
            seq, frame = got
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size)
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
            last = now

            tracks = tracker.update(detector.detect(frame), now)
            if face_worker is not None:
                for result in face_worker.results():
                    tracker.apply_face(result)
                due = tracker.due_for_face_check(now, fcfg["recheck_known_s"], fcfg["recheck_unknown_s"],
                                                 fcfg["min_person_height_px"])
                if due and face_worker.submit(frame, [(t.id, t.box) for t in due]):
                    for t in due:
                        t.last_face_check = now

            major = [t for t in tracks if t.last_seen == now
                     and t.height >= dcfg["min_height_frac"] * size[1]
                     and now - t.first_seen >= dcfg["min_presence_s"]]
            annotated = frame.copy()
            overlay.draw_tracks(annotated, tracks, now)
            recorder.push(now, frame, annotated, bool(major), [t.name for t in major])

            if display.enabled:
                warnings = []
                if not storage.available():
                    warnings.append("NO STORAGE - NOT RECORDING")
                if not timesync.ok():
                    warnings.append("CLOCK NOT SYNCED")
                screen = annotated.copy()
                overlay.draw_status(screen, datetime.datetime.now(), fps, recorder.event_active, warnings)
                if not display.show(screen):
                    log.info("quit from keyboard")
                    break
    except Exception:
        log.exception("DoorCam crashed")
        exit_code = 1
    finally:
        log.info("shutting down")
        recorder.stop()
        camera.stop()
        if face_worker is not None:
            face_worker.stop()
        display.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
