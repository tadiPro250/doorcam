"""Person detection: jetson-inference (TensorRT, GPU) on the Jetson, OpenCV HOG elsewhere."""
import logging

import cv2

log = logging.getLogger(__name__)


def _import_jetson():
    try:
        from jetson_inference import detectNet
        from jetson_utils import cudaFromNumpy
        return detectNet, cudaFromNumpy
    except ImportError:
        pass
    try:
        import jetson.inference
        import jetson.utils
        return jetson.inference.detectNet, jetson.utils.cudaFromNumpy
    except ImportError:
        return None, None


class PersonDetector(object):
    def __init__(self, cfg):
        detect_net, self._to_cuda = _import_jetson()
        if detect_net is not None:
            # First run downloads the model and builds a TensorRT engine (a few minutes).
            self._net = detect_net(cfg["network"], threshold=cfg["threshold"])
            self._person_id = next(i for i in range(self._net.GetNumClasses())
                                   if self._net.GetClassDesc(i) == "person")
            log.info("person detector: jetson-inference %s", cfg["network"])
        else:
            self._net = None
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            log.warning("jetson-inference not found, using slow OpenCV HOG person detector")

    def detect(self, frame):
        """Return person boxes as (x1, y1, x2, y2) in pixels."""
        h, w = frame.shape[:2]
        if self._net is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = self._net.Detect(self._to_cuda(rgb), overlay="none")
            boxes = [(d.Left, d.Top, d.Right, d.Bottom) for d in dets if d.ClassID == self._person_id]
        else:
            rects, _ = self._hog.detectMultiScale(frame, winStride=(8, 8), scale=1.05)
            boxes = [(x, y, x + rw, y + rh) for (x, y, rw, rh) in rects]
        return [(max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)) for x1, y1, x2, y2 in boxes]
