"""Liveness challenge: a random sequence of face and hand actions done on cue.

A photo can't blink or turn its head, and a recorded video can't know which actions will be
asked for. Backends (face_recognition/dlib on the Jetson, MediaPipe on a laptop) turn each
frame into FaceSignals and a hand gesture label; the logic here is plain numpy.

Image coordinates: x to the right, y down, camera image NOT mirrored. A person turning to
their own left moves their nose towards the right of the image (positive yaw).
"""
import collections
import random

import numpy as np

# ear: eye openness (eye aspect ratio), yaw: nose offset from the eye midpoint / eye distance
FaceSignals = collections.namedtuple("FaceSignals", "ear yaw")

# name -> (kind, prompt)
ACTIONS = collections.OrderedDict([
    ("blink_twice", ("face", "Blink twice")),
    ("turn_left", ("face", "Turn your head to your left")),
    ("turn_right", ("face", "Turn your head to your right")),
    ("thumbs_up", ("hand", "Show a thumbs up")),
    ("open_palm", ("hand", "Show your open palm")),
    ("one_finger", ("hand", "Raise one finger")),
    ("two_fingers", ("hand", "Raise two fingers")),
    ("three_fingers", ("hand", "Raise three fingers")),
    ("fist", ("hand", "Make a fist")),
])

DEFAULTS = {
    "actions": 2,           # actions per attempt
    "pool": list(ACTIONS),  # actions that may be asked for
    "settle_s": 0.8,        # looking at the camera before the first action (sets the baselines)
    "settle_timeout_s": 6.0,    # to look at the camera and be recognised
    "recognise_frames": 3,      # frames matched to the card owner before the greeting
    "face_window_s": 3.0,   # time allowed per face action
    "hand_window_s": 4.0,   # time allowed per hand action
    "hold_s": 0.4,          # a turn or gesture must be held this long
    "wrong_hold_s": 0.8,    # holding a different action this long fails the attempt
    "face_lost_s": 0.5,     # the face may vanish this long (detector dropouts)
    "blink_ratio": 0.65,    # eye counts as closed below this fraction of its open size
    "turn_yaw": 0.25,       # yaw change that counts as a turn
    "neutral_yaw": 0.12,
    "camera_mirrored": False,
    "require_identity": True,  # every step must include a frame matched to the card owner
    "min_identity_frames": 5,  # and at least this many matched frames over the whole attempt
}

DROPOUT_S = 0.25  # a held turn/gesture survives detector misses this short

_rng = random.SystemRandom()


def pick_challenge(count, pool=None, rng=_rng):
    """Random actions without repeats, always including at least one face action:
    a hand gesture alone proves a live hand, not a live face."""
    pool = list(pool or ACTIONS)
    face = [a for a in pool if ACTIONS[a][0] == "face"]
    if not face:
        raise ValueError("the liveness pool needs at least one face action")
    count = max(1, min(count, len(pool)))
    first = rng.choice(face)
    seq = [first] + rng.sample([a for a in pool if a != first], count - 1)
    rng.shuffle(seq)
    return seq


# ---- face signals ---------------------------------------------------------

def _ear(eye):
    """Eye aspect ratio from 6 eye points in dlib order (corner, top, top, corner, bottom, bottom)."""
    e = np.asarray(eye, float)
    width = np.linalg.norm(e[0] - e[3])
    if width <= 0:
        return 0.0
    return (np.linalg.norm(e[1] - e[5]) + np.linalg.norm(e[2] - e[4])) / (2.0 * width)


def _signals(eye_a, eye_b, nose):
    a, b = np.mean(np.asarray(eye_a, float), axis=0), np.mean(np.asarray(eye_b, float), axis=0)
    dist = np.linalg.norm(a - b)
    yaw = (nose[0] - (a[0] + b[0]) / 2.0) / dist if dist > 0 else 0.0
    return FaceSignals((_ear(eye_a) + _ear(eye_b)) / 2.0, float(yaw))


def signals_from_dlib(landmarks):
    """From one face_recognition.face_landmarks() dict (dlib 68 points)."""
    return _signals(landmarks["left_eye"], landmarks["right_eye"], landmarks["nose_bridge"][-1])


MP_EYES = ([33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380])
MP_NOSE_TIP = 1


def signals_from_mediapipe(points):
    """From MediaPipe Face Mesh landmarks as an (N, 2) array in pixels."""
    p = np.asarray(points, float)
    return _signals(p[MP_EYES[0]], p[MP_EYES[1]], p[MP_NOSE_TIP])


# ---- hand gestures --------------------------------------------------------

FINGERS = ((6, 8), (10, 12), (14, 16), (18, 20))  # (middle joint, tip): index, middle, ring, pinky


def classify_hand(points):
    """Gesture name from 21 hand points (MediaPipe / trt_pose_hand order) in pixels, or None
    when the hand is not clearly making one of the gestures."""
    p = np.asarray(points, float)[:, :2]

    def d(i, j):
        return np.linalg.norm(p[i] - p[j])

    palm = d(0, 9)
    if palm <= 0:
        return None
    fingers = []
    for joint, tip in FINGERS:
        ratio = d(tip, 0) / max(d(joint, 0), 1e-6)
        fingers.append(True if ratio > 1.2 else (False if ratio < 1.0 else None))
    if None in fingers:
        return None
    thumb_reach = d(4, 9) / palm
    thumb = True if thumb_reach > 0.8 else (False if thumb_reach < 0.6 else None)

    if fingers == [True, True, True, True]:
        return "open_palm" if thumb is not False else None
    if fingers == [True, False, False, False]:
        return "one_finger"
    if fingers == [True, True, False, False]:
        return "two_fingers"
    if fingers == [True, True, True, False]:
        return "three_fingers"
    if fingers == [False, False, False, False]:
        if thumb is False:
            return "fist"
        if thumb:
            v = p[4] - p[2]
            if v[1] < -0.6 * np.linalg.norm(v):  # pointing up
                return "thumbs_up"
    return None


# ---- the challenge --------------------------------------------------------

class Step(object):
    def __init__(self, action, start, cue_s):
        self.action = action
        self.kind, self.prompt = ACTIONS[action]
        self.start = start
        self.cue_s = cue_s         # how long the spoken prompt takes; the window starts after it
        self.done_at = None
        self.neutral_seen = False
        self.hold_start = None
        self.last_hit = None
        self.wrong = None          # (label, since)
        self.blinks = 0
        self.eyes_closed = False
        self.identity_seen = False


class Challenge(object):
    """Feed it every frame with update(); it reports "running", "passed" or "failed".

    Order: look at the camera -> recognised as the card owner (`recognised_at` is set: play the
    greeting, which lasts greeting_len s) -> the actions. prompt_len(action) gives each spoken
    prompt's length: an action's time limit starts when its prompt has finished. Play the
    prompt when `step` changes."""

    def __init__(self, cfg, now, actions=None, prompt_len=None, greeting_len=0.0):
        self.cfg = dict(DEFAULTS, **(cfg or {}))
        self.actions = actions or pick_challenge(self.cfg["actions"], self.cfg["pool"])
        self._prompt_len = prompt_len or (lambda action: 0.0)
        self.status = "running"
        self.reason = ""
        self.started = now
        self.recognised_at = None
        self._greeting_len = greeting_len
        self.steps = []
        self._settle = []          # (ear, yaw) samples
        self._settle_from = None
        self._last_face = None
        self.identity_frames = 0
        self.base_ear = self.base_yaw = None

    @property
    def step(self):
        return self.steps[-1] if self.steps and self.status == "running" else None

    @property
    def prompt(self):
        if self.status != "running":
            return ""
        return self.step.prompt if self.step else "Look at the camera"

    def _window(self, s):
        return s.cue_s + (self.cfg["face_window_s"] if s.kind == "face" else self.cfg["hand_window_s"])

    def time_left(self, now):
        s = self.step
        return None if s is None else max(0.0, self._window(s) - (now - s.start))

    def _next_step(self, now):
        action = self.actions[len(self.steps)]
        self.steps.append(Step(action, now, self._prompt_len(action)))

    def _fail(self, reason):
        self.status, self.reason = "failed", reason
        return self.status

    def update(self, now, n_faces, signals, gesture=None, identity_ok=None):
        """n_faces: faces in the frame; signals: FaceSignals of the face, or None;
        gesture: classify_hand() label or None; identity_ok: True/False if the face was
        compared with the card owner this frame, None if not checked."""
        if self.status != "running":
            return self.status
        cfg = self.cfg
        if n_faces > 1:
            return self._fail("more than one face")
        if identity_ok is False:
            return self._fail("face does not match the card")
        if signals is not None and n_faces == 1:
            self._last_face = now
        elif self._last_face is not None and now - self._last_face > cfg["face_lost_s"]:
            return self._fail("face left the camera")

        if identity_ok:
            self.identity_frames += 1
        if not self.steps:
            return self._update_settle(now, signals)

        s = self.step
        if identity_ok:
            s.identity_seen = True
        if signals is not None:
            hit = self._face_action(s, now, signals) if s.kind == "face" else \
                self._hand_action(s, now, gesture)
            if self.status != "running":
                return self.status
            if hit:
                last = len(self.steps) == len(self.actions)
                if cfg["require_identity"] and (not s.identity_seen or
                                                (last and self.identity_frames < cfg["min_identity_frames"])):
                    pass  # done, but wait (within the window) for enough identity checks
                else:
                    s.done_at = now
                    if last:
                        self.status = "passed"
                        return self.status
                    self._next_step(now)
                    return self.status
        if now - s.start > self._window(s):
            return self._fail("too slow: %s" % s.prompt.lower())
        return self.status

    def _update_settle(self, now, signals):
        cfg = self.cfg
        if signals is not None:
            if self._settle_from is None:
                self._settle_from = now
            self._settle.append(signals)
            if self.recognised_at is None and now - self._settle_from >= cfg["settle_s"] and \
                    len(self._settle) >= 3 and (not cfg["require_identity"] or
                                                self.identity_frames >= cfg["recognise_frames"]):
                self.base_ear = float(np.median([x.ear for x in self._settle]))
                self.base_yaw = float(np.median([x.yaw for x in self._settle]))
                self.recognised_at = now
        if self.recognised_at is not None:
            if now - self.recognised_at >= self._greeting_len:
                self._next_step(now)
            return self.status
        if now - self.started > cfg["settle_timeout_s"]:
            return self._fail("face not recognised" if self._settle else "no face in front of the camera")
        return self.status

    def _held(self, s, now, on):
        """True once `on` has been continuously true for hold_s (misses up to DROPOUT_S allowed)."""
        if on:
            if s.hold_start is None or now - s.last_hit > DROPOUT_S:
                s.hold_start = now
            s.last_hit = now
            return now - s.hold_start >= self.cfg["hold_s"]
        if s.last_hit is not None and now - s.last_hit > DROPOUT_S:
            s.hold_start = None
        return False

    def _wrong(self, s, now, label, what):
        """Fail if a different action is held for wrong_hold_s."""
        if label is None:
            s.wrong = None
            return
        if s.wrong is None or s.wrong[0] != label:
            s.wrong = (label, now)
        elif now - s.wrong[1] >= self.cfg["wrong_hold_s"]:
            self._fail("wrong action: %s instead of %s" % (what, s.prompt.lower()))

    def _face_action(self, s, now, signals):
        cfg = self.cfg
        yaw = signals.yaw - self.base_yaw
        if cfg["camera_mirrored"]:
            yaw = -yaw
        turned = "left" if yaw > cfg["turn_yaw"] else ("right" if yaw < -cfg["turn_yaw"] else None)
        if abs(yaw) < cfg["neutral_yaw"]:
            s.neutral_seen = True

        if s.action == "blink_twice":
            closed = signals.ear < cfg["blink_ratio"] * self.base_ear
            if s.eyes_closed and not closed:
                s.blinks += 1
            s.eyes_closed = closed
            self._wrong(s, now, turned, "turned " + str(turned))
            return s.blinks >= 2

        want = s.action.split("_")[1]
        self._wrong(s, now, turned if turned != want else None, "turned " + str(turned))
        return s.neutral_seen and self._held(s, now, turned == want)

    def _hand_action(self, s, now, gesture):
        if gesture != s.action:
            s.neutral_seen = True
        label = gesture if gesture not in (None, s.action) else None
        self._wrong(s, now, label, ACTIONS[label][1].lower() if label else "")
        return s.neutral_seen and self._held(s, now, gesture == s.action)
