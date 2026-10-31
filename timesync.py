"""Is the system clock trustworthy yet? Footage is only filed under a date once it is."""
import logging
import os
import subprocess
import time

log = logging.getLogger(__name__)
CHECK_EVERY_S = 30


class TimeSync(object):
    def __init__(self, cfg):
        self.required = cfg["require_sync"]
        self.flag = cfg["sync_flag"]
        self._ok = False
        self._checked = -CHECK_EVERY_S

    def ok(self):
        if not self.required or self._ok:
            return True  # once synced, chrony keeps it synced for the rest of this boot
        now = time.monotonic()
        if now - self._checked >= CHECK_EVERY_S:
            self._checked = now
            self._ok = os.path.exists(self.flag) or self._chrony_synced()
            if self._ok:
                log.info("system clock is synchronised")
        return self._ok

    @staticmethod
    def _chrony_synced():
        try:
            return subprocess.call(["chronyc", "waitsync", "1"], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=5) == 0
        except (OSError, subprocess.TimeoutExpired):
            return False
