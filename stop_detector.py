"""
stop_detector.py

Standalone stop-sign/marker detection module for DonkeyCar, using a
trained YOLO model (stop.pt) instead of (or alongside) color-based
detection.

Same interface pattern as HumanDetector so it can be dropped into
mylane.py the same way later, without touching lane-following logic.
Works on a copy of the frame only - never modifies the original image.
"""

import cv2

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(
        "ultralytics is required for StopDetector. "
        "Install with: pip install ultralytics"
    ) from e


class StopDetector:

    def __init__(self,
                 model_path="/home/pi/mycar/models/stop.pt",
                 conf_thresh=0.5,
                 infer_size=320,
                 frame_skip=1):
        """
        model_path : path to the trained YOLO weights (stop.pt)
        conf_thresh: minimum confidence to count as a valid detection
        infer_size : the COPY of the frame is resized to this square
                      size before inference (keeps it fast on a Pi)
        frame_skip : run inference every Nth frame (1 = every frame)
        """
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.infer_size = infer_size
        self.frame_skip = max(1, frame_skip)

        self._frame_count = 0
        self._last_detected = False

    def detect(self, img_arr):
        """
        img_arr: the ORIGINAL frame. Never modified in place; only a
                  copy is resized/annotated for inference.

        Returns:
            stop_detected (bool)
            annotated (np.ndarray or None) - image copy with boxes
                       drawn, for optional debug display
        """
        if img_arr is None:
            return False, None

        self._frame_count += 1

        if (self._frame_count - 1) % self.frame_skip != 0:
            return self._last_detected, None

        infer_img = cv2.resize(img_arr.copy(),
                                (self.infer_size, self.infer_size))

        results = self.model.predict(
            source=infer_img,
            conf=self.conf_thresh,
            verbose=False
        )

        detected = False
        annotated = None

        if results:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                detected = True
            annotated = r.plot()

        self._last_detected = detected
        return detected, annotated
