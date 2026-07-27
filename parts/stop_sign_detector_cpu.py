"""
CPU-based StopSignDetector skeleton for DonkeyCar.

NOTE:
This is a starting implementation. TensorFlow Lite object detection models
have different output tensor orders depending on the model. You may need to
adjust the output indices if your model differs.
"""

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter


class StopSignDetector:
    STOP_SIGN_CLASS_ID = 12

    def __init__(self,
                 min_score=0.5,
                 show_bounding_box=True,
                 max_reverse_count=0,
                 reverse_throttle=-0.5):

        self.min_score = min_score
        self.show_bounding_box = show_bounding_box
        self.max_reverse_count = max_reverse_count
        self.reverse_count = 0
        self.reverse_throttle = reverse_throttle
        self.is_reversing = False

        self.interpreter = Interpreter("models/detect.tflite")
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.h = self.input_details[0]["shape"][1]
        self.w = self.input_details[0]["shape"][2]

        with open("models/labelmap.txt") as f:
            self.labels = [l.strip() for l in f]

    def _preprocess(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.w, self.h))
        data = np.expand_dims(resized, 0)

        if self.input_details[0]["dtype"] == np.float32:
            data = (np.float32(data) - 127.5) / 127.5

        return data

    def detect_stop_sign(self, img):
        inp = self._preprocess(img)

        self.interpreter.set_tensor(self.input_details[0]["index"], inp)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]

        H, W = img.shape[:2]

        for box, cls, score in zip(boxes, classes, scores):
            if score < self.min_score:
                continue
            if int(cls) != self.STOP_SIGN_CLASS_ID:
                continue

            ymin, xmin, ymax, xmax = box
            xmin = int(xmin * W)
            xmax = int(xmax * W)
            ymin = int(ymin * H)
            ymax = int(ymax * H)

            if self.show_bounding_box:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                cv2.putText(img, f"Stop {score:.2f}",
                            (xmin, max(20, ymin-10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,0), 2)

            return True, img

        return False, img

    def run(self, img_arr, throttle):

        if img_arr is None:
            return throttle, img_arr

        found, img_arr = self.detect_stop_sign(img_arr)

        if found or self.is_reversing:

            if self.reverse_count < self.max_reverse_count:
                self.is_reversing = True
                self.reverse_count += 1
                return self.reverse_throttle, img_arr

            self.is_reversing = False
            return 0.0, img_arr

        self.is_reversing = False
        self.reverse_count = 0
        return throttle, img_arr

    def shutdown(self):
        pass
