"""Live face recognition for the camera view: finds faces on the newest frame in the background
and names them against everyone registered (faces_db). The display draws the latest result,
so the picture stays smooth even when recognition is slower than the camera.
"""
import logging
import threading
import time

import numpy as np

import verify

log = logging.getLogger(__name__)
PERSON_EVERY_S = 0.2  # person detection rate when the Jetson detector is available


class Recognizer(threading.Thread):
    def __init__(self, verifier, points, faces_db, person_detector=None):
        super().__init__(name="recognizer", daemon=True)
        self.verifier, self.points, self.person_detector = verifier, points, person_detector
        self.accept = verifier.cfg["accept"]
        self.min_eye_px = verifier.cfg["min_eye_px"]
        self.people = []  # (display name, signatures)
        for folder in faces_db.names():
            try:
                self.people.append((faces_db.display_name(folder), faces_db.load(folder)))
            except (IOError, OSError, ValueError):
                log.warning("could not load face data of %s", folder)
        self._cond = threading.Condition()
        self._frame = None
        self._halt = threading.Event()
        self.faces = []    # [(box, name or None, score)]
        self.persons = []  # [box]
        self.rate = 0.0    # recognitions per second
        self._last_person = 0.0

    def submit(self, frame):
        """Hand over the newest frame (older unprocessed frames are dropped)."""
        with self._cond:
            self._frame = frame
            self._cond.notify()

    def stop(self):
        self._halt.set()
        with self._cond:
            self._cond.notify_all()

    def identify(self, emb):
        best, best_score = None, -1.0
        for name, sigs in self.people:
            score = verify.similarity(sigs, emb)
            if score > best_score:
                best, best_score = name, score
        return (best if best_score >= self.accept else None), best_score

    def run(self):
        while not self._halt.is_set():
            with self._cond:
                while self._frame is None and not self._halt.is_set():
                    self._cond.wait(0.5)
                frame, self._frame = self._frame, None
            if frame is None:
                continue
            t0 = time.monotonic()
            try:
                found = []
                for f in self.points.faces(frame):
                    name, score = None, None
                    if np.linalg.norm(f.pts5[0] - f.pts5[1]) >= self.min_eye_px * 0.6:
                        face = verify.align(frame, f.pts5)
                        if face is not None:
                            name, score = self.identify(self.verifier.embed(face))
                    found.append((tuple(int(v) for v in f.box), name, score))
                self.faces = found
                if self.person_detector is not None and t0 - self._last_person >= PERSON_EVERY_S:
                    self.persons = self.person_detector.detect(frame)
                    self._last_person = t0
            except Exception:
                log.exception("live recognition failed on a frame")
            dt = time.monotonic() - t0
            self.rate = 0.8 * self.rate + 0.2 / max(dt, 1e-3)
