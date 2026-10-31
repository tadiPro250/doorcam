"""Face landmarks from whichever library is installed: MediaPipe (laptop) or dlib via
face_recognition (Jetson, Python 3.6). Both give the same Face: 5 points for alignment,
blink/turn signals for the liveness challenge, and head angles in degrees.

Angles: yaw + = the person turned to their own left, pitch + = looking up (camera image not mirrored).
"""
import collections

import cv2
import numpy as np

import liveness

Face = collections.namedtuple("Face", "box pts5 signals yaw pitch")

# A generic face in camera-style axes (x right, y down, z away from the camera), arbitrary units:
# nose tip, chin, image-left eye outer corner, image-right eye outer corner, mouth corners.
MODEL_3D = np.array([(0, 0, 0), (0, 330, 65), (-225, -170, 135), (225, -170, 135),
                     (-150, 150, 125), (150, 150, 125)], np.float64)


def head_angles(pts6, size):
    """(yaw, pitch) in degrees from the 6 points above, by fitting the 3D face model."""
    w, h = size
    cam = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1]], np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, np.asarray(pts6, np.float64), cam, np.zeros(4),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0
    rot, _ = cv2.Rodrigues(rvec)
    d = rot.dot([0.0, 0.0, -1.0])  # the direction the face points, in camera axes
    return float(np.degrees(np.arctan2(d[0], -d[2]))), float(np.degrees(np.arctan2(-d[1], -d[2])))


class MediaPipeBackend(object):
    name = "mediapipe"
    SIX = [1, 152, 33, 263, 61, 291]

    def __init__(self, max_faces=2):
        import mediapipe as mp
        self.mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=max_faces, min_detection_confidence=0.5)

    def faces(self, frame):
        h, w = frame.shape[:2]
        found = self.mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_face_landmarks or []
        out = []
        for lm in found:
            p = np.array([(l.x * w, l.y * h) for l in lm.landmark])
            pts5 = np.array([p[liveness.MP_EYES[0]].mean(0), p[liveness.MP_EYES[1]].mean(0),
                             p[1], p[61], p[291]], np.float32)
            yaw, pitch = head_angles(p[self.SIX], (w, h))
            (x1, y1), (x2, y2) = p.min(0), p.max(0)
            out.append(Face((x1, y1, x2, y2), pts5, liveness.signals_from_mediapipe(p), yaw, pitch))
        return out


class DlibBackend(object):
    name = "dlib"

    def __init__(self, detect_width=320, model="hog"):
        import face_recognition
        self.fr = face_recognition
        self.detect_width = detect_width
        self.model = model

    def faces(self, frame):
        h, w = frame.shape[:2]
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        scale = min(1.0, self.detect_width / float(w))
        small = cv2.resize(rgb, None, fx=scale, fy=scale) if scale < 1 else rgb
        locs = self.fr.face_locations(small, number_of_times_to_upsample=1, model=self.model)
        locs = [tuple(int(round(v / scale)) for v in loc) for loc in locs]
        out = []
        for (top, right, bottom, left), lm in zip(locs, self.fr.face_landmarks(rgb, locs)):
            nose, chin = lm["nose_bridge"][-1], lm["chin"][8]
            six = [nose, chin, lm["left_eye"][0], lm["right_eye"][3], lm["top_lip"][0], lm["top_lip"][6]]
            pts5 = np.array([np.mean(lm["left_eye"], 0), np.mean(lm["right_eye"], 0), nose,
                             lm["top_lip"][0], lm["top_lip"][6]], np.float32)
            yaw, pitch = head_angles(six, (w, h))
            out.append(Face((left, top, right, bottom), pts5, liveness.signals_from_dlib(lm), yaw, pitch))
        return out


def landmarker(prefer=None, max_faces=2):
    """The first backend that is installed (MediaPipe, then dlib), or None. dlib finds any number of faces."""
    order = [prefer] if prefer else ["mediapipe", "dlib"]
    for name in order:
        try:
            return MediaPipeBackend(max_faces) if name == "mediapipe" else DlibBackend()
        except ImportError:
            continue
    return None
