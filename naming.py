"""Turning a person's name into a folder / file name."""
import hashlib
import re


def safe_name(name):
    """Folder name for a person. Names with no Latin letters/digits (e.g. Devanagari) get a stable
    id instead of an empty name."""
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return s or "person_" + hashlib.md5(name.strip().encode("utf-8")).hexdigest()[:10]
