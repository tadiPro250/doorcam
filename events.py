"""SQLite log of person events (who, when, which clip). Lives next to the footage."""
import os
import sqlite3


class EventLog(object):
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY,
            start TEXT, end TEXT, seconds REAL,
            names TEXT, clip TEXT, snapshot TEXT)""")
        self.db.commit()

    def add(self, start, end, names, clip, snapshot):
        self.db.execute(
            "INSERT INTO events (start, end, seconds, names, clip, snapshot) VALUES (?, ?, ?, ?, ?, ?)",
            (start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"),
             round((end - start).total_seconds(), 1), ",".join(names), clip, snapshot))
        self.db.commit()

    def prune_missing(self):
        """Drop rows whose clip was deleted by retention."""
        rows = self.db.execute("SELECT id, clip FROM events").fetchall()
        gone = [(rid,) for rid, clip in rows if not os.path.exists(clip)]
        if gone:
            self.db.executemany("DELETE FROM events WHERE id = ?", gone)
            self.db.commit()

    def close(self):
        self.db.close()
