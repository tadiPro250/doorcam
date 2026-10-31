#!/usr/bin/env python3
"""Card registry: which card belongs to which registered face (users.db, SQLite).

A card tap looks up one row here; only that person's face data is then read from
faces_db/<folder>/, used for the one attempt, and dropped.

    users.py list
    users.py assign CARD_UID NAME [--admin]   # NAME must already be registered (register_face.py)
    users.py revoke CARD_UID                  # card stops working; face data is kept
    users.py restore CARD_UID
    users.py delete CARD_UID                  # remove the person: card, face data and greeting

Each person has exactly one card: assigning a new card to someone replaces their old one.
"""
import argparse
import collections
import os
import sqlite3
import sys
import time

import verify

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "users.db")

User = collections.namedtuple("User", "card name folder admin active added")


def normalise_uid(uid):
    """Readers print UIDs differently (04:A1:B2, 04a1b2, "04 A1 B2"); store one form."""
    return "".join(c for c in uid.upper() if c.isalnum())


class Users(object):
    def __init__(self, path=DB_PATH):
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS users (
            card TEXT PRIMARY KEY, name TEXT NOT NULL, folder TEXT NOT NULL,
            admin INTEGER NOT NULL DEFAULT 0, active INTEGER NOT NULL DEFAULT 1, added TEXT)""")
        self.db.commit()

    def lookup(self, uid):
        row = self.db.execute("SELECT card, name, folder, admin, active, added FROM users WHERE card = ?",
                              (normalise_uid(uid),)).fetchone()
        return self._user(row) if row else None

    @staticmethod
    def _user(row):
        return User(row[0], row[1], row[2], bool(row[3]), bool(row[4]), row[5] or "")

    def assign(self, uid, name, folder, admin=False):
        """Give the card to this person. Each person has exactly one card: any other card they had
        is removed, and the card stops belonging to whoever had it before."""
        uid = normalise_uid(uid)
        self.db.execute("DELETE FROM users WHERE folder = ? AND card <> ?", (folder, uid))
        self.db.execute("INSERT OR REPLACE INTO users (card, name, folder, admin, active, added) VALUES (?, ?, ?, ?, 1, ?)",
                        (uid, name, folder, int(admin), time.strftime("%Y-%m-%d %H:%M:%S")))
        self.db.commit()

    def card_of(self, folder):
        row = self.db.execute("SELECT card, name, folder, admin, active, added FROM users WHERE folder = ?",
                              (folder,)).fetchone()
        return self._user(row) if row else None

    def set_admin(self, uid, admin):
        self.db.execute("UPDATE users SET admin = ? WHERE card = ?", (int(admin), normalise_uid(uid)))
        self.db.commit()

    def delete_person(self, folder):
        self.db.execute("DELETE FROM users WHERE folder = ?", (folder,))
        self.db.commit()

    def set_active(self, uid, active):
        cur = self.db.execute("UPDATE users SET active = ? WHERE card = ?", (int(active), normalise_uid(uid)))
        self.db.commit()
        return cur.rowcount > 0

    def all(self):
        return [self._user(r) for r in
                self.db.execute("SELECT card, name, folder, admin, active, added FROM users ORDER BY name")]

    def admins(self):
        return [u for u in self.all() if u.admin and u.active]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list")
    a = sub.add_parser("assign")
    a.add_argument("card")
    a.add_argument("name")
    a.add_argument("--admin", action="store_true")
    for cmd in ("revoke", "restore", "delete"):
        sub.add_parser(cmd).add_argument("card")
    args = ap.parse_args()

    users = Users()
    faces = verify.FaceDB(os.path.join(APP_DIR, "faces_db"))
    if args.cmd == "assign":
        folder = verify.safe_name(args.name)
        if folder not in faces.names():
            print("%s has no registered face (run register_face.py first)" % args.name)
            return 1
        users.assign(args.card, faces.display_name(folder), folder, args.admin)
        print("card %s -> %s%s" % (normalise_uid(args.card), faces.display_name(folder), " (admin)" if args.admin else ""))
    elif args.cmd == "delete":
        user = users.lookup(args.card)
        if user is None:
            print("no such card")
            return 1
        if input("Delete %s (card %s, face data and greeting)? [y/N] " % (user.name, user.card)).strip().lower() != "y":
            return 1
        users.delete_person(user.folder)
        faces.remove(user.folder)
        greeting = os.path.join(APP_DIR, "prompts", "hello_%s.wav" % user.folder)
        if os.path.exists(greeting):
            os.remove(greeting)
        print("deleted %s" % user.name)
    elif args.cmd in ("revoke", "restore"):
        if not users.set_active(args.card, args.cmd == "restore"):
            print("no such card")
            return 1
        print("card %s %s" % (normalise_uid(args.card), "revoked" if args.cmd == "revoke" else "restored"))
    else:
        rows = users.all()
        for u in rows:
            print("%-16s %-20s %s%s" % (u.card, u.name, "admin " if u.admin else "", "" if u.active else "REVOKED"))
        if not rows:
            print("no cards assigned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
