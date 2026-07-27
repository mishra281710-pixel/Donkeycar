import cv2
import numpy as np
import time

class LaneFollower:

    def __init__(self, pid=None, cfg=None):

        self.pid = pid
        self.cfg = cfg

        self.prev_steering = 0.0
        self.stop_detected = False
        self.stop_until = 0
        self.base_throttle = 0.4

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

        height, width, _ = img_arr.shape

        # ======================================================
        # STOP MARKER DETECTION (RED)
        # ======================================================

                # ======================================================
        # TRAFFIC MARKER DETECTION
        # ======================================================

        stop_roi = img_arr[int(height * 0.70):height, :]
        hsv_marker = cv2.cvtColor(stop_roi, cv2.COLOR_RGB2HSV)

        # ---------------- RED ----------------
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        red_mask1 = cv2.inRange(hsv_marker, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv_marker, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        red_pixels = cv2.countNonZero(red_mask)

        # ---------------- GREEN ----------------
        lower_green = np.array([40, 80, 80])
        upper_green = np.array([85, 255, 255])

        green_mask = cv2.inRange(hsv_marker,
                                 lower_green,
                                 upper_green)

        green_pixels = cv2.countNonZero(green_mask)

        # ---------------- BLUE ----------------
        # ---------------- ORANGE ----------------
        lower_orange = np.array([8, 100, 100])
        upper_orange = np.array([22, 255, 255])

        orange_mask = cv2.inRange(hsv_marker,
                                  lower_orange,
                                  upper_orange)

        orange_pixels = cv2.countNonZero(orange_mask) 
        marker = None

        if red_pixels > 100:
            print("STOP DETECTED")
            self.stop_until = time.time() + 5
            return 0.0, 0.0, img_arr

        elif green_pixels > 100:
            print("GREEN DETECTED")
            marker = "move"

        elif orange_pixels > 100:
            print("ORANGE DETECTED")
            marker = "slow"

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

        
        # Normal lane-following throttle
        if abs(steering) > 0.40:
            throttle = 0.28

        elif abs(steering) > 0.20:
            throttle = 0.32

        else:
            throttle = 0.38

        # Override based on traffic markers
        if marker == "move":
            throttle = 0.50      # speed up

        elif marker == "slow":
            throttle = 0.20   

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
        if marker is not None:

            color = (255, 255, 255)

            if marker == "green":
                color = (0, 255, 0)

            elif marker == "blue":
                color = (255, 0, 0)

            cv2.putText(img_arr,
                        marker.upper(),
                        (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)
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
