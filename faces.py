"""Face recognition (dlib via face_recognition) on a worker thread, fed with person boxes only."""
import collections
import datetime
import logging
import os
import pickle
import threading
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
RELOAD_CHECK_S = 30

# person_box is where the person was when the face was checked
FaceResult = collections.namedtuple("FaceResult", "track_id name distance face_box person_box")

try:
    import face_recognition
except ImportError:
    face_recognition = None


def available():
    return face_recognition is not None


def find_faces(rgb, model="hog", upsample=1):
    """face_recognition locations (top, right, bottom, left); retries harder on small images."""
    locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=upsample, model=model)
    if not locs and upsample < 2 and min(rgb.shape[:2]) < 300:
        locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=2, model=model)
    return locs


def largest(locs):
    return max(locs, key=lambda l: (l[2] - l[0]) * (l[1] - l[3]))


def encode_image(path):
    """Encoding of the largest face in a photo, or None if no face is found."""
    rgb = face_recognition.load_image_file(path)
    locs = find_faces(rgb)
    if not locs:
        return None
    return face_recognition.face_encodings(rgb, [largest(locs)])[0]


class KnownFaces(object):
    """known_faces/<name>/*.jpg, with encodings cached so restarts are fast."""

    def __init__(self, folder):
        self.folder = folder
        self.cache_path = os.path.join(folder, ".encodings.pkl")
        self.names = []
        self.encodings = np.zeros((0, 128))
        self._signature = None

    def _scan(self):
        found = []
        if not os.path.isdir(self.folder):
            return found
        for name in sorted(os.listdir(self.folder)):
            person_dir = os.path.join(self.folder, name)
            if name.startswith(".") or not os.path.isdir(person_dir):
                continue
            for fn in sorted(os.listdir(person_dir)):
                if fn.lower().endswith(IMAGE_EXTS):
                    path = os.path.join(person_dir, fn)
                    found.append((name, path, os.path.getmtime(path)))
        return found

    def reload_if_changed(self):
        files = self._scan()
        signature = tuple(files)
        if signature == self._signature:
            return
        self._signature = signature
        try:
            with open(self.cache_path, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}
        names, encodings, new_cache = [], [], {}
        for name, path, mtime in files:
            cached = cache.get(path)
            enc = cached[1] if cached and cached[0] == mtime else encode_image(path)
            new_cache[path] = (mtime, enc)
            if enc is None:
                log.warning("no face found in %s, ignoring it", path)
                continue
            names.append(name)
            encodings.append(enc)
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(new_cache, f)
        except OSError as e:
            log.warning("could not write face cache: %s", e)
        self.names = names
        self.encodings = np.array(encodings) if encodings else np.zeros((0, 128))
        counts = collections.Counter(names)
        log.info("known faces: %s", ", ".join("%s (%d)" % kv for kv in sorted(counts.items())) or "none")

    def match(self, encoding, tolerance):
        if not self.names:
            return "Unknown", None
        dists = face_recognition.face_distance(self.encodings, encoding)
        i = int(np.argmin(dists))
        return (self.names[i] if dists[i] <= tolerance else "Unknown"), float(dists[i])


class FaceWorker(threading.Thread):
    """Takes one job at a time (the newest frame wins), so it never falls behind the camera."""

    def __init__(self, cfg, known_dir, storage):
        super().__init__(name="faces", daemon=True)
        self.cfg = cfg
        self.storage = storage
        self.known = KnownFaces(known_dir)
        self._cond = threading.Condition()
        self._job = None
        self._results = []
        self._unknown_saved = {}  # track_id -> (count, last_saved)
        self._halt = threading.Event()

    def stop(self):
        self._halt.set()
        with self._cond:
            self._cond.notify_all()

    def submit(self, frame, jobs):
        """jobs: [(track_id, person_box)]. Returns False if still busy with the previous frame."""
        with self._cond:
            if self._job is not None:
                return False
            self._job = (frame, jobs)
            self._cond.notify()
            return True

    def results(self):
        with self._cond:
            out, self._results = self._results, []
        return out

    def run(self):
        self.known.reload_if_changed()
        last_reload = time.monotonic()
        while not self._halt.is_set():
            with self._cond:
                if self._job is None:
                    self._cond.wait(1.0)
                job = self._job
            if job is not None:
                try:
                    results = self._process(*job)
                except Exception:
                    log.exception("face recognition failed")
                    results = []
                with self._cond:
                    self._results.extend(results)
                    self._job = None
            if time.monotonic() - last_reload > RELOAD_CHECK_S:
                self.known.reload_if_changed()
                last_reload = time.monotonic()

    def _process(self, frame, jobs):
        h, w = frame.shape[:2]
        results = []
        for track_id, (x1, y1, x2, y2) in jobs:
            # Faces are in the top part of a person box; searching only there is much faster.
            pad = int((x2 - x1) * 0.15)
            cx1, cx2 = max(0, int(x1) - pad), min(w, int(x2) + pad)
            cy1, cy2 = max(0, int(y1) - pad), min(h, int(y1 + (y2 - y1) * 0.55))
            if cx2 - cx1 < 40 or cy2 - cy1 < 40:
                continue
            rgb = np.ascontiguousarray(frame[cy1:cy2, cx1:cx2, ::-1])
            locs = find_faces(rgb, self.cfg["model"], self.cfg["upsample"])
            if not locs:
                results.append(FaceResult(track_id, None, None, None, None))
                continue
            top, right, bottom, left = largest(locs)
            enc = face_recognition.face_encodings(rgb, [(top, right, bottom, left)])[0]
            name, dist = self.known.match(enc, self.cfg["tolerance"])
            face_box = (cx1 + left, cy1 + top, cx1 + right, cy1 + bottom)
            results.append(FaceResult(track_id, name, dist, face_box, (x1, y1, x2, y2)))
            if name == "Unknown" and self.cfg["save_unknown"]:
                self._save_unknown(frame, track_id, face_box)
        return results

    def _save_unknown(self, frame, track_id, face_box):
        count, last = self._unknown_saved.get(track_id, (0, 0.0))
        now = time.monotonic()
        if count >= 3 or now - last < 5 or not self.storage.available():
            return
        if len(self._unknown_saved) > 1000:
            self._unknown_saved.clear()
        self._unknown_saved[track_id] = (count + 1, now)
        x1, y1, x2, y2 = face_box
        m = int((x2 - x1) * 0.4)
        h, w = frame.shape[:2]
        crop = frame[max(0, y1 - m):min(h, y2 + m), max(0, x1 - m):min(w, x2 + m)]
        wall = datetime.datetime.now()
        folder = self.storage.day_dir("unknown_faces", self.storage.day_name(wall))
        cv2.imwrite(os.path.join(folder, "%s_t%d.jpg" % (wall.strftime("%H%M%S"), track_id)), crop)
