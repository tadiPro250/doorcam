"""Spoken prompts for the outside speaker: pre-made WAV files in prompts/, played without blocking.

make_prompts.py creates the files. The person at the door never hears why an attempt failed,
only that it did.
"""
import logging
import os
import re
import subprocess
import sys
import time
import wave

import liveness
from naming import safe_name

log = logging.getLogger(__name__)

TEXTS = dict((name, prompt + ".") for name, (_, prompt) in liveness.ACTIONS.items())
TEXTS.update({
    "look_at_camera": "Please look at the camera.",
    "granted": "Access granted.",
    "denied": "Access denied. Please wait, someone will open the door.",
    "card_unknown": "Card not recognised.",
    "card_locked": "This card is locked. Please wait, someone will open the door.",
})


# Guidance spoken while registering a face (register_face.py shows the same words on screen).
REGISTRATION = [
    "Look straight at the camera", "Turn your head to your left", "Turn your head to your right",
    "Tilt your head up", "Tilt your head down",
    "Turn a little more left", "Turn a little more right", "Tilt a little more up", "Tilt a little more down",
    "Too far - come back a little", "Keep your head level", "Don't turn sideways", "Hold still", "Good",
    "Move into view", "Only one face, please", "Come closer", "Better lighting needed",
    "Registration complete", "Registration failed - please try again",
]


def reg_key(message):
    return "reg_" + re.sub(r"[^a-z0-9]+", "_", message.lower()).strip("_")


TEXTS.update((reg_key(m), m.replace(" - ", ", ") + ".") for m in REGISTRATION)


def hello_key(name):
    return "hello_" + safe_name(name)


def hello_text(name):
    return "Hi %s, please look at the camera." % name


class Speaker(object):
    def __init__(self, folder, device=""):
        self.folder = folder
        self.device = device  # ALSA device for aplay, e.g. "plughw:1,0"; "" = default
        self._proc = None
        self._lengths = {}
        self._win_until = 0.0  # Windows plays through winsound, which has no process to watch

    def path(self, key):
        return os.path.join(self.folder, key + ".wav")

    def missing(self):
        return [k for k in sorted(TEXTS) if not os.path.isfile(self.path(k))]

    def duration(self, key):
        """Length of a prompt in seconds (0 if the file is missing)."""
        if key not in self._lengths:
            try:
                w = wave.open(self.path(key), "rb")
                self._lengths[key] = w.getnframes() / float(w.getframerate())
                w.close()
            except (IOError, OSError, wave.Error):
                return 0.0
        return self._lengths[key]

    def play(self, key, fallback=None):
        """Start playing (cutting off whatever is playing). Returns the clip length in seconds."""
        if not os.path.isfile(self.path(key)) and fallback:
            key = fallback
        path = self.path(key)
        if not os.path.isfile(path):
            log.warning("missing prompt %s", path)
            return 0.0
        self.stop()
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            self._win_until = time.monotonic() + self.duration(key)
            return self.duration(key)
        if sys.platform == "darwin":
            cmd = ["afplay", path]
        else:
            cmd = ["aplay", "-q"] + (["-D", self.device] if self.device else []) + [path]
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as e:
            log.error("cannot play %s: %s", path, e)
        return self.duration(key)

    def busy(self):
        if sys.platform == "win32":
            return time.monotonic() < self._win_until
        return self._proc is not None and self._proc.poll() is None

    def stop(self):
        if sys.platform == "win32" and self._win_until:
            import winsound
            winsound.PlaySound(None, 0)
            self._win_until = 0.0
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None
