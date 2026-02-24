import cv2
import numpy as np

class LaneFollower:
    def __init__(self):
        self.prev_steering = 0.0

    def run(self, img_arr):

        height, width, _ = img_arr.shape
        roi = img_arr[int(height*0.6):height, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([20, 80, 80])
        upper_yellow = np.array([35, 255, 255])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.GaussianBlur(mask, (5,5), 0)

        ys, xs = np.where(mask > 0)

        steering = 0.0
        throttle = 0.0

        if len(xs) > 400:

            cx = np.mean(xs)

            # 🔥 Shift slightly left
             

            steering = ((cx - width/2) / (width/2)) 

            # Clamp steering between -1 and 1
            steering = max(-1.0, min(1.0, steering))

            # Smooth steering
            steering = 0.5*(0.7 * self.prev_steering + 0.3*steering) 
            self.prev_steering = steering

            throttle = 0.5

        else:
            throttle = 0.0

        return float(steering), float(throttle)

