"""Drawing: name boxes (also burned into event clips) and the on-screen status bar."""
import cv2
import numpy as np

from tracker import iou

GREEN = (80, 200, 80)    # recognised
RED = (60, 60, 230)      # face seen, not recognised
YELLOW = (0, 200, 255)   # person, no face seen yet
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
SHOW_FOR_S = 0.3  # keep a box on screen briefly if the detector misses a frame


def _label(img, text, x, y, color):
    (tw, th), base = cv2.getTextSize(text, FONT, 0.6, 2)
    y = max(y, th + base + 4)
    cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - base - 2), FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def draw_tracks(img, tracks, now):
    for t in tracks:
        if now - t.last_seen > SHOW_FOR_S:
            continue
        color = GREEN if t.known else (RED if t.face_seen else YELLOW)
        x1, y1, x2, y2 = [int(v) for v in t.box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        _label(img, t.name, x1, y1, color)
        # Only while the person is still where they were when the face was checked.
        if t.face_box is not None and iou(t.box, t.face_ref) > 0.8:
            fx1, fy1, fx2, fy2 = t.face_box
            cv2.rectangle(img, (fx1, fy1), (fx2, fy2), color, 1)


def draw_status(img, wall, fps, recording, warnings):
    w = img.shape[1]
    cv2.rectangle(img, (0, 0), (w, 24), (0, 0, 0), -1)
    cv2.putText(img, wall.strftime("%a %d %b %Y  %H:%M:%S") + "   %.0f fps" % fps, (6, 17),
                FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    x = w - 8
    if recording:
        cv2.circle(img, (x - 6, 12), 6, RED, -1)
        cv2.putText(img, "REC", (x - 46, 17), FONT, 0.5, RED, 1, cv2.LINE_AA)
        x -= 56
    for text in warnings:
        (tw, _), _ = cv2.getTextSize(text, FONT, 0.5, 1)
        x -= tw + 12
        cv2.putText(img, text, (x, 17), FONT, 0.5, YELLOW, 1, cv2.LINE_AA)


def message_frame(size, text):
    img = np.zeros((size[1], size[0], 3), np.uint8)
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.8, 2)
    cv2.putText(img, text, ((size[0] - tw) // 2, (size[1] + th) // 2), FONT, 0.8, WHITE, 2, cv2.LINE_AA)
    return img
