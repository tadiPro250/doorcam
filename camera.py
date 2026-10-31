"""USB webcam capture on a background thread that survives slow boots and unplugs."""
import logging
import os
import sys
import threading
import time

import cv2

log = logging.getLogger(__name__)


class Camera(threading.Thread):
    def __init__(self, cfg):
        super().__init__(name="camera", daemon=True)
        self.cfg = cfg
        self.connected = False
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._halt = threading.Event()

    def stop(self):
        self._halt.set()

    def read(self, last_seq, timeout=1.0):
        """Return (seq, frame) for a frame newer than last_seq, or None on timeout."""
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            if self._seq == last_seq:
                return None
            return self._seq, self._frame

    def run(self):
        while not self._halt.is_set():
            cap = self._open()
            if cap is None:
                self._halt.wait(2.0)
                continue
            self.connected = True
            failures = 0
            while not self._halt.is_set():
                ok, frame = cap.read()
                if not ok:
                    failures += 1
                    if failures >= 20:
                        log.warning("camera stopped delivering frames, reconnecting")
                        break
                    time.sleep(0.05)
                    continue
                failures = 0
                with self._cond:
                    self._frame = frame
                    self._seq += 1
                    self._cond.notify_all()
            cap.release()
            self.connected = False

    def _open(self):
        dev = self.cfg["device"]
        path = dev if isinstance(dev, str) else "/dev/video%d" % dev
        on_linux = sys.platform.startswith("linux")
        # At boot the USB camera can appear a few seconds after we start.
        if on_linux and not os.path.exists(path):
            return None
        # Force V4L2: OpenCV's default GStreamer backend gave the boot-time errors in cam-startup.log.
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2) if on_linux else cv2.VideoCapture(dev)
        if not cap.isOpened():
            cap.release()
            return None
        if self.cfg.get("fourcc"):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cfg["fourcc"]))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg["height"])
        cap.set(cv2.CAP_PROP_FPS, self.cfg["fps"])
        log.info("camera %s opened at %dx%d @ %.0f fps", path,
                 cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                 cap.get(cv2.CAP_PROP_FPS))
        return cap
