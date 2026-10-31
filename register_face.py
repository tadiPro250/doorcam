#!/usr/bin/env python3
"""Register a person's face from the webcam (the admin app does the same on the Jetson).

    python3 register_face.py "Ravi Kumar"   (Jetson: uses dlib)
    uv run --python 3.11 --with mediapipe==0.10.14 --with "numpy<2" --with opencv-python \\
        --with pyyaml --with onnxruntime --with piper-tts python register_face.py "Ravi Kumar"   (laptop)

Guided like Face ID with voice prompts (see registration.py). Saves the face signatures to
faces_db/ and makes the "Hi <name>" greeting. Q/ESC cancels.
"""
import argparse
import os
import sys
import time
import warnings

os.environ.setdefault("GLOG_minloglevel", "2")  # quieter MediaPipe start-up
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2
import yaml

import facepoints
import registration
import verify
from prompts import Speaker
from registration import LEVEL_DEG, STRAIGHT_DEG, TARGET_DEG

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, YELLOW, WHITE, GREY = (80, 200, 80), (0, 200, 255), (255, 255, 255), (150, 150, 150)


def text(img, s, y, scale=0.8, color=WHITE, x=None):
    (tw, _), _ = cv2.getTextSize(s, FONT, scale, 2)
    x = (img.shape[1] - tw) // 2 if x is None else x
    cv2.putText(img, s, (x, y), FONT, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, color, 2, cv2.LINE_AA)


def draw_guide(img, guide, in_zone):
    """Face ID-style dial: the dot is where the head points, the box is where it should go.
    The screen is a mirror, so turning to your left moves the dot left."""
    h, w = img.shape[:2]
    r, s = 70, 70.0 / 40  # radius, pixels per degree
    cx, cy = w - r - 20, r + 80
    cv2.circle(img, (cx, cy), r, (40, 40, 40), -1)
    cv2.circle(img, (cx, cy), r, GREY, 1, cv2.LINE_AA)
    cv2.line(img, (cx - r, cy), (cx + r, cy), GREY, 1)
    cv2.line(img, (cx, cy - r), (cx, cy + r), GREY, 1)
    if guide.finished:
        return
    _, axis, direction, _ = guide.pose
    lo, hi = TARGET_DEG
    if axis is None:
        box = (-STRAIGHT_DEG, -STRAIGHT_DEG, STRAIGHT_DEG, STRAIGHT_DEG)
    elif axis == "yaw":  # screen x = -yaw (mirror)
        a, b = sorted((-direction * lo, -direction * hi))
        box = (a, -LEVEL_DEG, b, LEVEL_DEG)
    else:  # screen y = -pitch
        a, b = sorted((-direction * lo, -direction * hi))
        box = (-LEVEL_DEG, a, LEVEL_DEG, b)
    x1, y1, x2, y2 = [int(round(v * s)) for v in box]
    cv2.rectangle(img, (cx + x1, cy + y1), (cx + x2, cy + y2), GREEN if in_zone else YELLOW, 2)
    rel = guide.relative()
    if rel is not None:
        dx = int(np.clip(-rel[0] * s, -r, r))
        dy = int(np.clip(-rel[1] * s, -r, r))
        cv2.circle(img, (cx + dx, cy + dy), 7, GREEN if in_zone else WHITE, -1, cv2.LINE_AA)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--quiet", action="store_true", help="no voice prompts")
    args = ap.parse_args()
    name = args.name.strip()
    if not name:
        sys.exit("give the person's name")

    with open(os.path.join(APP_DIR, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    verifier = verify.Verifier(cfg.get("verify"), APP_DIR)
    points = facepoints.landmarker()
    if points is None:
        sys.exit("needs MediaPipe (laptop) or face_recognition/dlib (Jetson)")
    speaker = None if args.quiet else Speaker(os.path.join(APP_DIR, "prompts"))
    if speaker is not None and speaker.missing():
        print("no audio for: %s (run make_prompts.py)" % ", ".join(speaker.missing()))
    if sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("cannot open the camera (is DoorCam running? quit it with Q first)")

    session = registration.Session(verifier, speaker)
    try:
        while not session.finished and not session.failed:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed")
                return 1
            session.feed(frame, points.faces(frame), time.monotonic())
            h, w = frame.shape[:2]
            view = cv2.flip(frame, 1)
            guide = session.guide
            if not guide.finished:
                text(view, guide.pose[0], 45, 0.9, YELLOW)
            text(view, session.msg, 85, 0.75, GREEN if session.in_zone or session.msg == "Good" else WHITE)
            draw_guide(view, guide, session.in_zone)
            cv2.rectangle(view, (20, 110), (20 + int(200 * session.progress), 120),
                          GREEN if session.in_zone else YELLOW, -1)
            cv2.rectangle(view, (20, 110), (220, 120), GREY, 1)
            cv2.rectangle(view, (0, h - 30), (int(w * len(session.faces) / float(session.total)), h), GREEN, -1)
            text(view, "%d / %d" % (len(session.faces), session.total), h - 9, 0.55)
            cv2.imshow("Register " + name, view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                print("cancelled")
                return 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
    if session.failed:
        print(session.failed)
        return 1

    other, score = session.lookalike(name)
    if other and input("This face looks like %s (%.2f), who is already registered. Save anyway? [y/N] "
                       % (other, score)).strip().lower() != "y":
        return 1
    ok, message = session.save(name)
    print(message)
    if ok:
        registration.make_greeting(name, lambda p: print("greeting: %s" % (p or "not made (no Piper)"))).join()
        while speaker is not None and speaker.busy():
            time.sleep(0.05)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
