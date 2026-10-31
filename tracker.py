"""Keeps an identity on each person box between (slow) face checks."""
import collections


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


class Track(object):
    def __init__(self, tid, box, now):
        self.id = tid
        self.box = box
        self.first_seen = now
        self.last_seen = now
        self.last_face_check = 0.0
        self.face_seen = False
        self.face_box = None
        self.face_ref = None
        self.votes = collections.Counter()  # known names matched on this track

    @property
    def known(self):
        return bool(self.votes)

    @property
    def name(self):
        if self.votes:
            return self.votes.most_common(1)[0][0]
        return "Unknown" if self.face_seen else "Person"

    @property
    def height(self):
        return self.box[3] - self.box[1]


class Tracker(object):
    def __init__(self, cfg):
        self.iou_match = cfg["iou_match"]
        self.max_missing = cfg["max_missing_s"]
        self.tracks = collections.OrderedDict()
        self._next_id = 1

    def update(self, boxes, now):
        """Match this frame's boxes to existing tracks (greedy by overlap). Returns live tracks."""
        pairs = sorted(((iou(t.box, b), tid, bi) for tid, t in self.tracks.items()
                        for bi, b in enumerate(boxes)), reverse=True)
        used_tracks, used_boxes = set(), set()
        for score, tid, bi in pairs:
            if score < self.iou_match:
                break
            if tid in used_tracks or bi in used_boxes:
                continue
            track = self.tracks[tid]
            track.box = boxes[bi]
            track.last_seen = now
            used_tracks.add(tid)
            used_boxes.add(bi)
        for bi, box in enumerate(boxes):
            if bi not in used_boxes:
                self.tracks[self._next_id] = Track(self._next_id, box, now)
                self._next_id += 1
        for tid in [tid for tid, t in self.tracks.items() if now - t.last_seen > self.max_missing]:
            del self.tracks[tid]
        return list(self.tracks.values())

    def due_for_face_check(self, now, recheck_known, recheck_unknown, min_height):
        due = []
        for t in self.tracks.values():
            if t.last_seen != now or t.height < min_height:
                continue
            interval = recheck_known if t.known else recheck_unknown
            if now - t.last_face_check >= interval:
                due.append(t)
        return due

    def apply_face(self, result):
        track = self.tracks.get(result.track_id)
        if track is None or result.name is None:
            return
        track.face_seen = True
        track.face_box = result.face_box
        track.face_ref = result.person_box
        if result.name != "Unknown":
            track.votes[result.name] += 1
