#!/usr/bin/env python3
"""Download the two models DoorCam needs into models/ (about 330 MB of downloads, once).

    python scripts/download_models.py

- Face recognition: ArcFace w600k_r50.onnx from InsightFace's buffalo_l pack
  (licensed for non-commercial research use, so it is downloaded rather than included).
- Voice: Piper en_US-lessac-medium, used to make the spoken prompts.
"""
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
ARCFACE_ZIP = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
VOICE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"


def fetch(url, dest):
    print("downloading %s" % url)
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write("\r  %3d%%  %d / %d MB" % (100 * done // total, done >> 20, total >> 20))
                sys.stdout.flush()
    print()
    os.replace(tmp, dest)


def main():
    os.makedirs(MODELS, exist_ok=True)
    face = os.path.join(MODELS, "w600k_r50.onnx")
    if os.path.exists(face):
        print("face model already there")
    else:
        zpath = os.path.join(tempfile.gettempdir(), "buffalo_l.zip")
        fetch(ARCFACE_ZIP, zpath)
        with zipfile.ZipFile(zpath) as z:
            name = [n for n in z.namelist() if n.endswith("w600k_r50.onnx")][0]
            with z.open(name) as src, open(face, "wb") as dst:
                shutil.copyfileobj(src, dst)
        os.remove(zpath)
        print("saved %s" % face)
    for ext in (".onnx", ".onnx.json"):
        dest = os.path.join(MODELS, "en_US-lessac-medium" + ext)
        if os.path.exists(dest):
            print("%s already there" % os.path.basename(dest))
        else:
            fetch(VOICE.replace(".onnx", ext), dest)
    print("models ready in %s" % MODELS)


if __name__ == "__main__":
    main()
