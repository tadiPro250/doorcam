#!/usr/bin/env python3
"""Add or remove people DoorCam should recognise. The running app picks up changes within ~30 s.

  python3 enroll.py Alice photo1.jpg photo2.jpg some_folder/
  python3 enroll.py Alice /footage/unknown_faces/2026-10-30_Fri/101530_t12.jpg
  python3 enroll.py Alice --camera        (quit DoorCam first: press Q on the monitor)
  python3 enroll.py --list
  python3 enroll.py --remove Alice

3-5 clear, front-facing photos per person in different lighting work best.
"""
import argparse
import datetime
import fcntl
import os
import shutil
import sys

import cv2

import faces

APP_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_DIR = os.path.join(APP_DIR, "known_faces")


def image_files(paths):
    for p in paths:
        if os.path.isdir(p):
            for fn in sorted(os.listdir(p)):
                if fn.lower().endswith(faces.IMAGE_EXTS):
                    yield os.path.join(p, fn)
        else:
            yield p


def add_photos(name, paths):
    dest = os.path.join(KNOWN_DIR, name)
    os.makedirs(dest, exist_ok=True)
    added = 0
    for path in image_files(paths):
        rgb = faces.face_recognition.load_image_file(path)
        count = len(faces.find_faces(rgb))
        if count != 1:
            print("skip %s: found %d faces (need exactly 1)" % (path, count))
            continue
        target = os.path.join(dest, os.path.basename(path))
        if os.path.exists(target):
            stem, ext = os.path.splitext(target)
            target = "%s_%s%s" % (stem, datetime.datetime.now().strftime("%H%M%S%f"), ext)
        shutil.copy2(path, target)
        added += 1
        print("added %s" % target)
    print("%d photo(s) added for %s" % (added, name))


def doorcam_running():
    with open("/tmp/doorcam.lock", "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
    return False


def add_from_camera(name, device, wanted=5):
    if doorcam_running():
        sys.exit("DoorCam is using the camera. Press Q on the monitor to quit it, then run this again.")
    dest = os.path.join(KNOWN_DIR, name)
    os.makedirs(dest, exist_ok=True)
    cap = cv2.VideoCapture(device)
    taken = 0
    print("Look at the camera. SPACE = take photo (when the box is green), Q = finish.")
    while taken < wanted:
        ok, frame = cap.read()
        if not ok:
            sys.exit("could not read from the camera")
        small = cv2.resize(frame, None, fx=0.5, fy=0.5)
        locs = faces.find_faces(small[:, :, ::-1].copy())
        view = frame.copy()
        color = (80, 200, 80) if len(locs) == 1 else (60, 60, 230)
        for top, right, bottom, left in locs:
            cv2.rectangle(view, (left * 2, top * 2), (right * 2, bottom * 2), color, 2)
        cv2.putText(view, "%s: %d/%d  SPACE=capture Q=done" % (name, taken, wanted), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("enroll", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" ") and len(locs) == 1:
            path = os.path.join(dest, "cam_%s.jpg" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            cv2.imwrite(path, frame)
            taken += 1
            print("saved %s" % path)
    cap.release()
    cv2.destroyAllWindows()
    print("%d photo(s) added for %s" % (taken, name))


def list_people():
    if not os.path.isdir(KNOWN_DIR):
        print("nobody enrolled yet")
        return
    for name in sorted(os.listdir(KNOWN_DIR)):
        folder = os.path.join(KNOWN_DIR, name)
        if os.path.isdir(folder) and not name.startswith("."):
            n = len([f for f in os.listdir(folder) if f.lower().endswith(faces.IMAGE_EXTS)])
            print("%-20s %d photo(s)" % (name, n))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", nargs="?")
    parser.add_argument("photos", nargs="*")
    parser.add_argument("--camera", action="store_true", help="take photos with the webcam")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--remove", metavar="NAME")
    args = parser.parse_args()

    if args.list:
        return list_people()
    if args.remove:
        folder = os.path.join(KNOWN_DIR, args.remove)
        if not os.path.isdir(folder):
            sys.exit("%s is not enrolled" % args.remove)
        if input("Delete all photos of %s? [y/N] " % args.remove).lower() == "y":
            shutil.rmtree(folder)
            print("removed %s" % args.remove)
        return
    if not args.name or not (args.photos or args.camera):
        parser.print_help()
        return
    if not faces.available():
        sys.exit("face_recognition is not installed (run setup.sh)")
    if args.camera:
        add_from_camera(args.name, args.device)
    else:
        add_photos(args.name, args.photos)


if __name__ == "__main__":
    main()
