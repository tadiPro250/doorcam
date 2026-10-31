#!/usr/bin/env python3
"""Try the entry check on a laptop webcam: face match + liveness challenge (MediaPipe points).

    uv run --python 3.11 --with mediapipe==0.10.14 --with "numpy<2" --with opencv-python \\
        --with pyyaml --with onnxruntime python liveness_demo.py --card CARD_UID

SPACE = tap that card, U = tap an unregistered card, Q/ESC = quit. Cards are linked to
registered faces with users.py. On a tap, only the card owner's face data is loaded from
disk; it is dropped again when the attempt ends. Prompts are spoken from prompts/ and also
shown on screen unless --hide-prompts.
"""
import argparse
import os
import time
import warnings

os.environ.setdefault("GLOG_minloglevel", "2")  # quieter MediaPipe start-up
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2
import mediapipe as mp
import numpy as np
import yaml

import liveness
import prompts
import verify
from prompts import Speaker
from users import Users

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, RED, WHITE, YELLOW = (80, 200, 80), (60, 60, 230), (255, 255, 255), (0, 200, 255)


def text(img, s, y, scale=0.8, color=WHITE, thick=2):
    (tw, _), _ = cv2.getTextSize(s, FONT, scale, thick)
    x = (img.shape[1] - tw) // 2
    cv2.putText(img, s, (x, y), FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def hand_covers_face(hand_pts, face_pts):
    """A hand in front of the face spoils the match score, so such frames aren't judged."""
    if hand_pts is None:
        return False
    (hx1, hy1), (hx2, hy2) = hand_pts.min(0), hand_pts.max(0)
    (fx1, fy1), (fx2, fy2) = face_pts.min(0), face_pts.max(0)
    return hx1 < fx2 and hx2 > fx1 and hy1 < fy2 and hy2 > fy1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--actions", type=int, help="actions per challenge (default from config.yaml)")
    ap.add_argument("--hide-prompts", action="store_true", help="speaker only, like at the door")
    ap.add_argument("--card", help="card UID that SPACE taps (default: the only assigned card)")
    args = ap.parse_args()

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(cfg_path) as f:
        full_cfg = yaml.safe_load(f)
    cfg = dict(full_cfg.get("liveness") or {})
    verifier = verify.Verifier(full_cfg.get("verify"), os.path.dirname(cfg_path))
    users = Users()
    cards = users.all()
    card = args.card or (cards[0].card if len(cards) == 1 else None)
    if card is None:
        print("cards: %s. Pick one with --card UID (assign cards with users.py)." %
              (", ".join("%s=%s" % (u.card, u.name) for u in cards) or "none assigned"))
        return 1
    signatures, owner_name = None, None   # only while an attempt is running
    if args.actions:
        cfg["actions"] = args.actions

    speaker = Speaker(os.path.join(os.path.dirname(cfg_path), "prompts"))
    if speaker.missing():
        print("no audio for: %s (run make_prompts.py)" % ", ".join(speaker.missing()))

    cap = cv2.VideoCapture(args.camera)
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=2, min_detection_confidence=0.5)
    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6)

    challenge, result_until, spoken, greeted = None, 0.0, 0, False
    card_msg, card_msg_until = "", 0.0
    match = ""
    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera read failed")
            break
        now = time.monotonic()
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces_lm = face_mesh.process(rgb).multi_face_landmarks or []
        hand_lm = hands.process(rgb).multi_hand_landmarks or []

        signals = pts = None
        if len(faces_lm) == 1:
            pts = np.array([(l.x * w, l.y * h) for l in faces_lm[0].landmark])
            signals = liveness.signals_from_mediapipe(pts)
        gesture = None
        hand_pts = None
        if hand_lm:
            hand_pts = np.array([(l.x * w, l.y * h) for l in hand_lm[0].landmark])
            gesture = liveness.classify_hand(hand_pts)

        identity_ok = None
        if signatures is not None and pts is not None and challenge is not None and challenge.status == "running":
            identity_ok, score, why = verifier.check(frame, verify.five_points_from_mediapipe(pts), signals.yaw,
                                                     signatures, covered=hand_covers_face(hand_pts, pts))
            match = ("match %.2f" % score if score is not None else "match -") + ("  " + why if why else "")

        if challenge is not None and challenge.status == "running":
            if challenge.update(now, len(faces_lm), signals, gesture, identity_ok) != "running":
                result_until = now + 3.0
                speaker.play("granted" if challenge.status == "passed" else "denied")
                print("%s: %s %s" % (challenge.status.upper(), " -> ".join(challenge.actions), challenge.reason))
                signatures = None  # the owner's face data is not kept after the attempt
            elif challenge.recognised_at is not None and not greeted:
                greeted = True
                if owner_name:
                    speaker.play(prompts.hello_key(owner_name))
            elif len(challenge.steps) > spoken:
                spoken = len(challenge.steps)
                speaker.play(challenge.step.action)

        # The decision uses the unmirrored frame; the screen is mirrored so it feels natural.
        view = cv2.flip(frame, 1)
        if hand_pts is not None:
            for x, y in hand_pts:
                cv2.circle(view, (int(w - x), int(y)), 3, YELLOW, -1)
        cv2.rectangle(view, (0, h - 28), (w, h), (0, 0, 0), -1)
        debug = "faces %d   gesture %s   %s" % (len(faces_lm), gesture or "-", match)
        if signals is not None:
            debug += "   eye %.2f   yaw %+.2f" % (signals.ear, signals.yaw)
        cv2.putText(view, debug, (8, h - 9), FONT, 0.5, WHITE, 1, cv2.LINE_AA)

        if challenge is not None and challenge.status == "running":
            n = len(challenge.steps)
            if challenge.recognised_at is not None and not n and owner_name:
                text(view, "Hi %s" % owner_name, 50, 1.0, GREEN)
            elif not args.hide_prompts:
                text(view, challenge.prompt, 50, 1.0, YELLOW)
            if n:
                text(view, "step %d of %d   %.1f s" % (n, len(challenge.actions), challenge.time_left(now)), 85, 0.6)
        elif now < card_msg_until:
            text(view, card_msg, 60, 1.0, RED)
        elif challenge is not None and now < result_until:
            passed = challenge.status == "passed"
            text(view, "PASSED" if passed else "FAILED", 60, 1.4, GREEN if passed else RED, 3)
            if not passed:
                text(view, challenge.reason, 100, 0.6)
        else:
            text(view, "SPACE = tap card %s    U = unregistered card" % card, 50, 0.7)

        cv2.imshow("Liveness demo", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        idle = challenge is None or challenge.status != "running"
        if key in (ord(" "), ord("u")) and idle:
            uid = card if key == ord(" ") else "UNREGISTERED"
            user = users.lookup(uid)
            if user is None or not user.active:
                speaker.play("card_unknown" if user is None else "card_locked")
                card_msg = "Card not recognised" if user is None else "Card revoked"
                card_msg_until, challenge = now + 2.5, None
                print("card %s: %s" % (uid, card_msg.lower()))
                continue
            # Load only this person's face data, for this attempt only.
            t0 = time.monotonic()
            try:
                signatures, owner_name = verifier.db.load(user.folder), user.name
            except (IOError, OSError):
                speaker.play("card_unknown")
                card_msg, card_msg_until, challenge = "No face registered for this card", now + 2.5, None
                print("card %s: %s has no face data (register them again)" % (uid, user.name))
                continue
            print("card %s -> %s: loaded %d face signatures (%.1f KB) in %.1f ms" %
                  (user.card, user.name, len(signatures), signatures.nbytes / 1024.0, (time.monotonic() - t0) * 1000))
            cfg["require_identity"] = True
            challenge = liveness.Challenge(cfg, now, prompt_len=speaker.duration,
                                           greeting_len=speaker.duration(prompts.hello_key(owner_name)))
            spoken, match, greeted = 0, "", False
            speaker.play("look_at_camera")
            print("challenge: " + " -> ".join(challenge.actions))
    speaker.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
