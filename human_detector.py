"""
human_detector.py

Standalone human-detection module for DonkeyCar.

Wraps a YOLO model (human.pt) and exposes a single detect() method.
This module is completely independent of the lane-following logic and
NEVER modifies the image that is handed to it - all resizing/inference
work happens on a copy.
"""

import cv2

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(
        "ultralytics is required for HumanDetector. "
        "Install with: pip install ultralytics"
    ) from e


class HumanDetector:

    def __init__(self,
                 model_path="/home/pi/mycar/models/human.pt",
                 conf_thresh=0.5,
                 infer_size=320,
                 frame_skip=1):
        """
        model_path : path to the trained YOLO weights (human.pt)
        conf_thresh: minimum confidence to count as a valid detection
        infer_size : the COPY of the frame is resized to this square
                      size before inference, to keep things fast on a
                      Raspberry Pi. Does not affect the original image.
        frame_skip : run inference every Nth frame (1 = every frame).
                      On skipped frames, the previous result is reused.
                      Increase this if the Pi cannot keep up with the
                      camera frame rate.
        """
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.infer_size = infer_size
        self.frame_skip = max(1, frame_skip)

        self._frame_count = 0
        self._last_detected = False

    def detect(self, img_arr):
        """
        img_arr: the ORIGINAL frame. This method never writes to it and
                  never resizes it in place - only a copy is touched.

        Returns:
            human_detected (bool)
            annotated (np.ndarray or None) - a separate image with boxes
                       drawn, useful for debugging/display. None if no
                       inference was run on this call (frame skipped) or
                       if img_arr was None.
        """
        if img_arr is None:
            return False, None

        self._frame_count += 1

        # Frame skipping to save CPU on the Pi - reuse last known result
        if (self._frame_count - 1) % self.frame_skip != 0:
            return self._last_detected, None

        # Work on a COPY only - the caller's array is never touched
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
            annotated = r.plot()  # returns a brand-new array

        self._last_detected = detected
        return detected, annotated
