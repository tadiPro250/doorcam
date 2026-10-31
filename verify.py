"""Entry face check: ArcFace signatures (InsightFace w600k_r50) compared 1:1 with the card owner.

Faces are straightened with 5 points (eyes, nose tip, mouth corners) before the model sees
them. The points can come from dlib (Jetson) or MediaPipe (laptop). Only good frames are
judged: large enough, facing the camera, sharp, evenly lit, nothing covering the face.

Note: the InsightFace model weights are licensed for non-commercial research use.
"""
import glob
import logging
import os
import shutil
import time

import cv2
import numpy as np

from naming import safe_name

log = logging.getLogger(__name__)

# Where the 5 points sit in the 112x112 image ArcFace was trained on (image-left eye first).
ARCFACE_REF = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)

DEFAULTS = {
    "model": "models/w600k_r50.onnx",
    "db_dir": "faces_db",
    "accept": 0.50,       # similarity at or above: this frame is the card owner
    "reject": 0.25,       # below: a different person, the attempt fails at once
    "min_eye_px": 40,     # distance between the eyes in pixels (face big enough)
    "max_yaw": 0.35,      # judge only frames facing the camera
    "min_sharpness": 40,  # variance of the Laplacian on the aligned face
    "brightness": [50, 210],
}


def five_points_from_mediapipe(points):
    p = np.asarray(points, np.float32)
    return np.array([p[[33, 160, 158, 133, 153, 144]].mean(0), p[[362, 385, 387, 263, 373, 380]].mean(0),
                     p[1], p[61], p[291]], np.float32)


def five_points_from_dlib(lm):
    """From one face_recognition.face_landmarks() dict."""
    return np.array([np.mean(lm["left_eye"], 0), np.mean(lm["right_eye"], 0), lm["nose_bridge"][-1],
                     lm["top_lip"][0], lm["top_lip"][6]], np.float32)


def align(frame, pts5):
    m, _ = cv2.estimateAffinePartial2D(np.asarray(pts5, np.float32), ARCFACE_REF, method=cv2.LMEDS)
    if m is None:
        return None
    return cv2.warpAffine(frame, m, (112, 112), borderValue=0)


class Embedder(object):
    def __init__(self, model_path):
        import onnxruntime as ort
        wanted = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in wanted if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input = self.session.get_inputs()[0].name
        log.info("face model %s on %s", os.path.basename(model_path), self.session.get_providers()[0])

    def __call__(self, face112_bgr):
        blob = (face112_bgr[:, :, ::-1].astype(np.float32) - 127.5) / 127.5
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])
        emb = self.session.run(None, {self.input: blob})[0][0]
        return emb / np.linalg.norm(emb)


class FaceDB(object):
    """faces_db/<name>/signatures.npy (+ the face crops they came from)."""

    def __init__(self, folder):
        self.folder = folder

    def names(self):
        if not os.path.isdir(self.folder):
            return []
        return sorted(n for n in os.listdir(self.folder)
                      if os.path.isfile(os.path.join(self.folder, n, "signatures.npy")))

    def folder_for(self, name):
        return safe_name(name)

    def exists(self, name):
        return os.path.isfile(os.path.join(self.folder, safe_name(name), "signatures.npy"))

    def load(self, folder):
        """Signatures of one person (folder name as listed by names())."""
        return np.load(os.path.join(self.folder, folder, "signatures.npy"))

    def display_name(self, folder):
        try:
            with open(os.path.join(self.folder, folder, "name.txt")) as f:
                return f.read().strip() or folder
        except (IOError, OSError):
            return folder

    def save(self, name, signatures, crops):
        folder = os.path.join(self.folder, safe_name(name))
        os.makedirs(folder, exist_ok=True)
        for old in glob.glob(os.path.join(folder, "*.jpg")):  # re-registering replaces the old crops
            os.remove(old)
        np.save(os.path.join(folder, "signatures.npy"), np.asarray(signatures, np.float32))
        with open(os.path.join(folder, "name.txt"), "w") as f:
            f.write(name.strip() + "\n")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for i, crop in enumerate(crops):
            cv2.imwrite(os.path.join(folder, "%s_%02d.jpg" % (stamp, i)), crop)
        return folder

    def remove(self, folder):
        shutil.rmtree(os.path.join(self.folder, folder), ignore_errors=True)


def similarity(signatures, emb):
    """Mean of the 3 best matches against the stored signatures (steadier than the single best)."""
    sims = np.sort(np.dot(signatures, emb))[::-1]
    return float(sims[:3].mean())


class Verifier(object):
    """check() returns (identity_ok, score, why): identity_ok is True/False, or None when the
    frame isn't good enough to judge."""

    def __init__(self, cfg, app_dir):
        self.cfg = dict(DEFAULTS, **(cfg or {}))
        path = lambda p: p if os.path.isabs(p) else os.path.join(app_dir, p)
        self.embed = Embedder(path(self.cfg["model"]))
        self.db = FaceDB(path(self.cfg["db_dir"]))

    def quality(self, frame, pts5, yaw, covered=False):
        """The aligned face if this frame is good enough to judge, else (None, reason)."""
        cfg = self.cfg
        if covered:
            return None, "covered"
        if np.linalg.norm(pts5[0] - pts5[1]) < cfg["min_eye_px"]:
            return None, "too far"
        if abs(yaw) > cfg["max_yaw"]:
            return None, "turned"
        face = align(frame, pts5)
        if face is None:
            return None, "align"
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < cfg["min_sharpness"]:
            return None, "blurry"
        lo, hi = cfg["brightness"]
        if not lo <= gray.mean() <= hi:
            return None, "lighting"
        return face, ""

    def check(self, frame, pts5, yaw, signatures, covered=False):
        face, why = self.quality(frame, pts5, yaw, covered)
        if face is None:
            return None, None, why
        score = similarity(signatures, self.embed(face))
        if score >= self.cfg["accept"]:
            return True, score, ""
        if score < self.cfg["reject"]:
            return False, score, "different person"
        return None, score, "unsure"
