import cv2
import numpy as np
import time

from stop_detector import StopDetector


class LaneFollower:

    def __init__(self, pid=None, cfg=None):

        self.pid = pid
        self.cfg = cfg

        self.prev_steering = 0.0
        self.stop_detected = False
        self.stop_until = 0
        self.base_throttle = 0.4

        # ---- NEW: human detector (separate, self-contained) ----
        # Only touches a COPY of the frame internally; never affects
        # img_arr or any of the lane-following logic below.
        self.stop_detector = StopDetector(
            model_path="/home/mishr/mycar/models/stop.pt"
        )

    def run(self, img_arr):

        if img_arr is None:
            return 0.0, 0.0, img_arr
        if time.time() < self.stop_until:
            return 0.0, 0.0, img_arr

        # -----------------------------
        # Already stopped
        # -----------------------------
        if self.stop_detected:
            cv2.putText(img_arr,
                        "STOPPED",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2)

            return 0.0, 0.0, img_arr

        # ======================================================
        # NEW: HUMAN DETECTION
        # Runs on a copy internally (see human_detector.py). img_arr
        # itself is untouched here, so nothing downstream is affected
        # unless a human is actually found.
        # ======================================================
        stop_detected, _ = self.stop_detector.detect(img_arr)

        if stop_detected:
            cv2.putText(img_arr,
                        "Stop Sign",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2)
            return 0.0, 0.0, img_arr

        height, width, _ = img_arr.shape

        # ======================================================
        # STOP MARKER DETECTION (RED)
        # ======================================================

        # ======================================================
        # LANE DETECTION
        # ======================================================

        roi = img_arr[int(height * 0.75):height, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

        lower_yellow = np.array([18, 60, 60])
        upper_yellow = np.array([45, 255, 255])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        M = cv2.moments(mask)

        steering = self.prev_steering

        if M["m00"] > 0:

            cx = int(M["m10"] / M["m00"])

            error = -(cx - (width // 2))

            steering = -float(error) / (width // 2)

            steering = max(min(steering, 1.0), -1.0)

        # Smooth steering
        steering = 0.7 * self.prev_steering + 0.7 * steering

        self.prev_steering = steering

        # ======================================================
        # THROTTLE CONTROL
        # ======================================================

        throttle = self.base_throttle

        if abs(steering) > 0.40:
            throttle = 0.28

        elif abs(steering) > 0.20:
            throttle = 0.32

        else:
            throttle = 0.38

        # ======================================================
        # DISPLAY
        # ======================================================

        cx = width // 2

        cv2.line(
            roi,
            (cx, 0),
            (cx, roi.shape[0]),
            (255, 0, 0),
            2
        )

        cv2.putText(img_arr,
                    f"Steering: {steering:.2f}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1)

        cv2.putText(img_arr,
                    f"Throttle: {throttle:.2f}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1)

        img_arr[int(height * 0.75):height, :] = cv2.cvtColor(
            mask,
            cv2.COLOR_GRAY2BGR
        )

        return steering, throttle, img_arr
