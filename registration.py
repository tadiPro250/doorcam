"""Face registration shared by register_face.py (laptop) and the admin app (Jetson).

Guided like Face ID: it first learns where "straight ahead" is for this person, then for each
direction says how far to turn and only takes frames once the head is held still in the
target zone. Works with any facepoints backend (MediaPipe or dlib).
"""
import os
import shutil
import subprocess
import threading
import time

import numpy as np

import prompts

APP_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE = os.path.join(APP_DIR, "models", "en_US-lessac-medium.onnx")

POSES = [  # (instruction, axis, direction, frames); yaw + = the person's left, pitch + = up
    ("Look straight at the camera", None, 0, 8),
    ("Turn your head to your left", "yaw", 1, 3),
    ("Turn your head to your right", "yaw", -1, 3),
    ("Tilt your head up", "pitch", 1, 3),
    ("Tilt your head down", "pitch", -1, 3),
]
TARGET_DEG = (15, 35)  # how far from straight ahead counts as turned (target zone)
LEVEL_DEG = 12         # the other direction must stay within this while turning
STRAIGHT_DEG = 7       # "straight" means within this of the person's own straight ahead
BASELINE_S = 1.0       # watching the person look straight to learn their straight ahead
HOLD_S = 0.6           # in the target zone, and steady, this long before frames are taken
STEADY_DEG = 3         # "steady": the head drifted less than this over the hold time
GAP_S = 0.2            # between frames
SMOOTH = 0.4           # weight of the newest angle (the estimate jitters a few degrees)
DONE_PAUSE_S = 0.8
POSE_TIMEOUT_S = 25
MIN_KEPT = 15
CONSISTENCY = 0.5      # every frame must look like the same person as the average


class PoseGuide(object):
    """Turns smoothed head angles into guidance, and says when a frame may be taken."""

    def __init__(self):
        self.pose_i = 0
        self.taken = 0
        self.base = None
        self._samples = []
        self._baseline_from = None
        self._angles = None
        self._zone_since = None
        self._recent = []          # (time, smoothed angles) over the last HOLD_S
        self._last_take = 0.0
        self.pose_start = None
        self.done_until = 0.0

    @property
    def finished(self):
        return self.pose_i >= len(POSES)

    @property
    def pose(self):
        return POSES[self.pose_i]

    def relative(self):
        if self.base is None or self._angles is None:
            return None
        return self._angles[0] - self.base[0], self._angles[1] - self.base[1]

    def lost(self):
        self._angles = None
        self._zone_since = None
        self._recent = []

    def _steady(self, now):
        self._recent = [(t, a) for t, a in self._recent if now - t <= HOLD_S] + [(now, self._angles.copy())]
        if now - self._recent[0][0] < HOLD_S * 0.75:
            return False
        # compare the start and end of the hold time (averaged, so jitter doesn't count as movement)
        a = np.array([x for _, x in self._recent])
        k = max(1, len(a) // 3)
        return bool(np.abs(a[-k:].mean(0) - a[:k].mean(0)).max() < STEADY_DEG)

    def update(self, now, yaw, pitch):
        """Returns (message, progress 0..1 towards the zone, ready_to_take)."""
        if self.pose_start is None:
            self.pose_start = now
        a = np.array([yaw, pitch])
        self._angles = a if self._angles is None else SMOOTH * a + (1 - SMOOTH) * self._angles
        if now < self.done_until:
            return "Good", 1.0, False

        if self.base is None:  # learn this person's straight ahead first
            if self._baseline_from is None:
                self._baseline_from = now
            self._samples.append(a)
            if now - self._baseline_from < BASELINE_S:
                return "Look straight at the camera", 0.0, False
            self.base = np.median(self._samples, axis=0)
            self._angles = self.base.copy()

        ry, rp = self.relative()
        _, axis, direction, _ = self.pose
        if axis is None:
            in_zone = abs(ry) <= STRAIGHT_DEG and abs(rp) <= STRAIGHT_DEG
            msg, progress = ("Hold still", 1.0) if in_zone else ("Look straight at the camera", 0.0)
        else:
            value, other = (ry, rp) if axis == "yaw" else (rp, ry)
            value *= direction
            lo, hi = TARGET_DEG
            in_zone = False
            progress = max(0.0, min(1.0, value / lo))
            word = {("yaw", 1): "left", ("yaw", -1): "right", ("pitch", 1): "up", ("pitch", -1): "down"}[(axis, direction)]
            verb = "Turn" if axis == "yaw" else "Tilt"
            if abs(other) > LEVEL_DEG:
                msg = "Keep your head level" if axis == "yaw" else "Don't turn sideways"
            elif value < lo:
                msg = "%s a little more %s" % (verb, word) if value > lo / 2 else self.pose[0]
            elif value > hi:
                msg = "Too far - come back a little"
            else:
                in_zone, msg = True, "Hold still"

        if not in_zone:
            self._zone_since = None
            return msg, progress, False
        if self._zone_since is None:
            self._zone_since = now
        steady = self._steady(now)
        ready = steady and now - self._zone_since >= HOLD_S and now - self._last_take >= GAP_S
        return msg, 1.0, ready

    def took(self, now):
        self.taken += 1
        self._last_take = now
        if self.taken >= self.pose[3]:
            self.pose_i += 1
            self.taken = 0
            self._zone_since = None
            self.pose_start = now
            self.done_until = now + DONE_PAUSE_S

    def timed_out(self, now):
        return self.pose_start is not None and now - self.pose_start > POSE_TIMEOUT_S


class VoiceCoach(object):
    """Speaks the guidance without chattering: a message is spoken once it has been showing for
    SETTLE_S, never over another prompt, and repeated only if nothing changes for REPEAT_S."""
    SETTLE_S = 0.5
    GAP_S = 0.4
    REPEAT_S = 5.0

    def __init__(self, speaker):
        self.speaker = speaker
        self._shown, self._shown_since = None, 0.0
        self._said, self._said_at, self._ended = None, -1e9, -1e9
        self._next = None  # spoken as soon as the speaker is free (a new pose's instruction)

    def queue(self, message):
        self._next = message

    def say(self, message, now):
        self.speaker.play(prompts.reg_key(message))
        self._said, self._said_at = message, now

    def update(self, message, now):
        if message != self._shown:
            self._shown, self._shown_since = message, now
        if self.speaker.busy():
            self._ended = now
            return
        if now - self._ended < self.GAP_S:
            return
        if self._next is not None:
            self.say(self._next, now)
            self._next = None
            return
        if now - self._shown_since < self.SETTLE_S:
            return
        if message != self._said or now - self._said_at >= self.REPEAT_S:
            self.say(message, now)

    def finish(self, message):
        """Say a last message and wait for it to end."""
        self.speaker.play(prompts.reg_key(message))
        while self.speaker.busy():
            time.sleep(0.05)


QUALITY_MSG = {"blurry": "Hold still", "too far": "Come closer", "lighting": "Better lighting needed"}
LOOKALIKE = 0.45  # a new face this similar to someone already registered needs a second look


class Session(object):
    """One registration. Feed every frame with feed(); when `finished`, call save(name)."""

    def __init__(self, verifier, speaker=None):
        self.verifier = verifier
        self.guide = PoseGuide()
        self.coach = VoiceCoach(speaker) if speaker is not None else None
        self.faces = []
        self.total = sum(p[3] for p in POSES)
        self.msg, self.progress, self.in_zone = POSES[0][0], 0.0, False
        self.failed = None
        self._sigs = None
        if self.coach:
            self.coach.queue(POSES[0][0])

    @property
    def finished(self):
        return self.guide.finished

    def feed(self, frame, faces, now):
        """faces: the facepoints.Face list for this frame."""
        g = self.guide
        if g.finished or self.failed:
            return
        self.progress, self.in_zone = 0.0, False
        if len(faces) != 1:
            g.lost()
            self.msg = "Only one face, please" if faces else "Move into view"
        else:
            f = faces[0]
            self.msg, self.progress, ready = g.update(now, f.yaw, f.pitch)
            self.in_zone = self.msg == "Hold still"
            if ready:
                # the guide has checked the head angle, so the verifier's own angle limit is skipped
                face, why = self.verifier.quality(frame, f.pts5, 0.0)
                if face is not None:
                    self.faces.append(face)
                    before = g.pose_i
                    g.took(now)
                    if g.pose_i != before and self.coach:
                        self.coach.say("Good", now)
                        if not g.finished:
                            self.coach.queue(g.pose[0])
                else:
                    self.msg = QUALITY_MSG.get(why, "Hold still")
        if g.timed_out(now):
            self.failed = "Timed out on: %s (%s)" % (g.pose[0].lower(), self.msg.lower())
            self.say_last("Registration failed - please try again")
        elif not g.finished and self.coach:
            self.coach.update(self.msg, now)

    def say_last(self, message):
        if self.coach:
            self.coach.speaker.play(prompts.reg_key(message))

    def signatures(self):
        """(signatures kept, crops kept, similarity of the worst frame) - computed once."""
        if self._sigs is None:
            sigs = np.array([self.verifier.embed(f) for f in self.faces])
            mean = sigs.mean(0)
            mean /= np.linalg.norm(mean)
            sims = sigs.dot(mean)
            keep = sims >= CONSISTENCY
            self._sigs = (sigs[keep], [f for f, k in zip(self.faces, keep) if k], float(sims.min()), mean)
        return self._sigs

    def lookalike(self, name):
        """(display name, score) of an already registered different person this face resembles."""
        _, _, _, mean = self.signatures()
        db, mine = self.verifier.db, self.verifier.db.folder_for(name)
        best = (None, 0.0)
        for folder in db.names():
            if folder == mine:
                continue
            score = float(np.max(db.load(folder).dot(mean)))
            if score > best[1]:
                best = (db.display_name(folder), score)
        return best if best[1] >= LOOKALIKE else (None, best[1])

    def save(self, name):
        """Returns (ok, message)."""
        sigs, crops, worst, _ = self.signatures()
        if len(sigs) < MIN_KEPT:
            self.say_last("Registration failed - please try again")
            return False, ("Only %d of %d frames looked like the same person (need %d). Make sure only "
                           "one person is in view and try again." % (len(sigs), len(self.faces), MIN_KEPT))
        self.verifier.db.save(name, sigs, crops)
        self.say_last("Registration complete")
        return True, "Registered %s with %d face signatures." % (name, len(sigs))


def make_greeting(name, done=None):
    """Make the "Hi <name>" prompt in the background (Piper package, else the piper program).
    done(path or None) is called when it finishes."""
    out = os.path.join(APP_DIR, "prompts", prompts.hello_key(name) + ".wav")

    def work():
        path = None
        try:
            import make_prompts
            try:
                synth = make_prompts.synth_package(VOICE)
            except ImportError:
                exe = shutil.which("piper")
                synth = make_prompts.synth_binary(VOICE, exe) if exe and os.path.exists(VOICE) else None
            if synth is not None:
                synth(prompts.hello_text(name), out)
                path = out
        except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError):
            path = None
        if done:
            done(path)
    t = threading.Thread(target=work, name="greeting", daemon=True)
    t.start()
    return t
