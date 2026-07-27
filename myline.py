import cv2
import numpy as np
import time

class LineFollower:
    def __init__(self, pid=None, cfg=None):
        self.prev_steering = 0.0
        self.base_throttle = 0.40
        self.stop_until = 0.0

    def run(self, img_arr):
        if img_arr is None:
            return 0.0, 0.0, None

        if time.time() < self.stop_until:
            return 0.0, 0.0, img_arr

        h, w, _ = img_arr.shape

        # ----- Red stop marker detection -----
        hsv_stop = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
        lower_red1 = np.array([0,80,80])
        upper_red1 = np.array([10,255,255])
        lower_red2 = np.array([170,80,80])
        upper_red2 = np.array([180,255,255])

        mask1 = cv2.inRange(hsv_stop, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_stop, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        red_pixels = cv2.countNonZero(red_mask)

        if red_pixels > 500:
            print("STOP DETECTED")
            self.stop_until = time.time() + 5
            return 0.0, 0.0, img_arr

        # ----- Yellow line detection -----
        roi = img_arr[int(h*0.70):h, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

        lower_yellow = np.array([18,80,80])
        upper_yellow = np.array([40,255,255])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((5,5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        steering = self.prev_steering
        throttle = self.base_throttle

        if contours:
            largest = max(contours, key=cv2.contourArea)

            if cv2.contourArea(largest) > 100:
                M = cv2.moments(largest)

                if M["m00"] > 0:
                    cx = int(M["m10"]/M["m00"])
                    error = cx - (w//2)

                    steering = (error/(w/2))
                    steering = max(min(steering,1.0),-1.0)
                    steering = 0.7*self.prev_steering + 0.3*steering
                    self.prev_steering = steering

                    if abs(steering) > 0.5:
                        throttle = 0.40
                    elif abs(steering) > 0.2:
                        throttle = 0.45
                    else:
                        throttle = 0.55

                    cv2.circle(roi, (cx, roi.shape[0]//2), 5, (0,255,0), -1)
        else:
            throttle = 0.0

        img_arr[int(h*0.70):h, :] = roi

        cv2.putText(img_arr, f"Steer:{steering:.2f}", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(img_arr, f"Throttle:{throttle:.2f}", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        return steering, throttle, img_arr
