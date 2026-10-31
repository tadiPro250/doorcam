"""Footage folders on the USB card, business-day retention and the low-space guard.

Layout:  <footage_dir>/<kind>/<YYYY-MM-DD_Ddd>/...   kind = continuous | events | unknown_faces
Footage recorded before the clock is synced goes to <kind>/unsynced/ and is deleted first.
"""
import datetime
import logging
import os
import re
import shutil
import time

log = logging.getLogger(__name__)

KINDS = ("continuous", "events", "unknown_faces")
UNSYNCED = "unsynced"
DAY_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_[A-Za-z]{3}$")
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
GB = 1024 ** 3


class Storage(object):
    def __init__(self, cfg, timesync):
        self.cfg = cfg
        self.root = os.path.abspath(cfg["footage_dir"])
        self.timesync = timesync
        self._ok = None
        self._checked = 0.0

    def available(self):
        now = time.monotonic()
        if self._ok is None or now - self._checked > 15:
            self._checked = now
            if not self.cfg["require_mount"]:
                os.makedirs(self.root, exist_ok=True)
            ok = (os.path.isdir(self.root) and os.access(self.root, os.W_OK)
                  and (os.path.ismount(self.root) or not self.cfg["require_mount"]))
            if ok != self._ok:
                if ok:
                    log.info("footage storage ready at %s", self.root)
                else:
                    log.error("footage storage NOT available at %s (card missing or not mounted)", self.root)
            self._ok = ok
        return self._ok

    def day_name(self, wall):
        if not self.timesync.ok():
            return UNSYNCED
        return "%s_%s" % (wall.strftime("%Y-%m-%d"), WEEKDAYS[wall.weekday()])

    def day_dir(self, kind, day):
        path = os.path.join(self.root, kind, day)
        os.makedirs(path, exist_ok=True)
        return path

    def enforce(self, protect=()):
        if not self.available():
            return
        protect = set(os.path.abspath(p) for p in protect if p)
        self._retention(protect)
        self._cap_events(protect)
        self._free_space(protect)

    # -- helpers ---------------------------------------------------------

    def _dated_dirs(self):
        """{date: [folder, ...]} across all kinds."""
        out = {}
        for kind in KINDS:
            base = os.path.join(self.root, kind)
            if not os.path.isdir(base):
                continue
            for name in os.listdir(base):
                m = DAY_RE.match(name)
                if m:
                    day = datetime.datetime.strptime(m.group(1), "%Y-%m-%d").date()
                    out.setdefault(day, []).append(os.path.join(base, name))
        return out

    def _retention(self, protect):
        """Keep the newest N business days that have footage; weekends go with their week."""
        dated = self._dated_dirs()
        if not self.timesync.ok():
            return  # can't tell which day is "today" yet
        today = datetime.date.today()
        business = sorted((d for d in dated if d.weekday() < 5 and d <= today), reverse=True)
        keep = self.cfg["keep_business_days"]
        if len(business) <= keep:
            return
        cutoff = business[keep - 1]
        for day in sorted(dated):
            if day >= cutoff:
                break
            for folder in dated[day]:
                if any(p.startswith(folder + os.sep) for p in protect):
                    continue
                log.info("retention: deleting %s", folder)
                shutil.rmtree(folder, ignore_errors=True)

    def _files(self, kinds):
        """All footage files, oldest first: unsynced folders, then by day, then by mtime."""
        files = []
        for kind in kinds:
            base = os.path.join(self.root, kind)
            if not os.path.isdir(base):
                continue
            for name in os.listdir(base):
                m = DAY_RE.match(name)
                if name == UNSYNCED:
                    rank = ""
                elif m:
                    rank = m.group(1)
                else:
                    continue
                folder = os.path.join(base, name)
                for fn in os.listdir(folder):
                    path = os.path.join(folder, fn)
                    try:
                        st = os.stat(path)
                    except OSError:
                        continue
                    files.append((rank, st.st_mtime, path, st.st_size))
        files.sort()
        return files

    def _delete(self, path):
        try:
            os.remove(path)
        except OSError as e:
            log.warning("could not delete %s: %s", path, e)
            return
        folder = os.path.dirname(path)
        try:
            if not os.listdir(folder):
                os.rmdir(folder)
        except OSError:
            pass

    def _cap_events(self, protect):
        files = self._files(["events"])
        total = sum(f[3] for f in files)
        limit = self.cfg["events_max_gb"] * GB
        for _, _, path, size in files:
            if total <= limit:
                break
            if path in protect:
                continue
            log.info("event space cap: deleting %s", path)
            self._delete(path)
            total -= size

    def _free_space(self, protect):
        limit = self.cfg["min_free_gb"] * GB
        if shutil.disk_usage(self.root).free >= limit:
            return
        for _, _, path, _ in self._files(KINDS):
            if path in protect:
                continue
            log.warning("low space: deleting %s", path)
            self._delete(path)
            if shutil.disk_usage(self.root).free >= limit:
                return
