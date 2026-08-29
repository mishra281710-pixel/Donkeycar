import cv2
import numpy as np
import time


class LaneFollower:

    def __init__(self, pid=None, cfg=None):

        self.pid = pid
        self.cfg = cfg

        self.prev_steering = 0.0
        self.base_throttle = 0.4

        # ArUco settings
        self.marker_actions = {
            0: "stop",
            1: "left",
            2: "right",
            3: "reverse"
        }

        self.action_time = {
            "stop": 3.0,
            "left": 1.2,
            "right": 1.2,
            "reverse": 1.5
        }

        self.action_command = {
            "stop": (0.0, 0.0),
            "left": (-0.85, 0.30),
            "right": (0.85, 0.30),
            "reverse": (0.0, -0.30)
        }

        self.current_action = None
        self.action_end = 0
        self.cooldown_end = 0

        # Ignore very small markers
        self.min_marker_area = 1500

        # ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )

        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(
                self.aruco_dict,
                self.aruco_params
            )
            self.new_aruco = True

        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.aruco_detector = None
            self.new_aruco = False


    def find_aruco(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if self.new_aruco:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.aruco_params
            )

        if ids is None:
            return None, corners, ids

        # Find the largest valid marker
        best_id = None
        best_area = 0

        for corner, marker_id in zip(corners, ids.flatten()):

            if marker_id not in self.marker_actions:
                continue

            points = corner.reshape(4, 2)
            area = cv2.contourArea(points)

            if area > self.min_marker_area and area > best_area:
                best_area = area
                best_id = int(marker_id)

        return best_id, corners, ids


    def run(self, img_arr):

        if img_arr is None:
            return 0.0, 0.0, img_arr

        now = time.time()

        # --------------------------------------------------
        # Continue an existing ArUco action
        # --------------------------------------------------

        if self.current_action is not None:

            if now < self.action_end:

                steering, throttle = self.action_command[
                    self.current_action
                ]

                cv2.putText(
                    img_arr,
                    "ARUCO: " + self.current_action.upper(),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2
                )

                return steering, throttle, img_arr

            # Action finished
            self.current_action = None
            self.cooldown_end = now + 2.0


        # --------------------------------------------------
        # Look for a new ArUco marker
        # --------------------------------------------------

        if now >= self.cooldown_end:

            marker_id, corners, ids = self.find_aruco(img_arr)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(
                    img_arr,
                    corners,
                    ids
                )

            if marker_id is not None:

                action = self.marker_actions[marker_id]

                print(
                    f"ArUco {marker_id} detected -> {action}"
                )

                self.current_action = action
                self.action_end = (
                    now + self.action_time[action]
                )

                steering, throttle = self.action_command[action]

                return steering, throttle, img_arr


        # --------------------------------------------------
        # Normal lane following
        # --------------------------------------------------

        height, width, _ = img_arr.shape

        roi = img_arr[int(height * 0.75):height, :]

        hsv = cv2.cvtColor(
            roi,
            cv2.COLOR_RGB2HSV
        )

        lower_yellow = np.array([18, 60, 60])
        upper_yellow = np.array([45, 255, 255])

        mask = cv2.inRange(
            hsv,
            lower_yellow,
            upper_yellow
        )

        moments = cv2.moments(mask)

        steering = self.prev_steering

        if moments["m00"] > 0:

            lane_x = int(
                moments["m10"] / moments["m00"]
            )

            error = lane_x - (width // 2)

            steering = -error / (width // 2)

            steering = max(
                -1.0,
                min(1.0, steering)
            )

        # Smooth steering
        steering = (
            0.7 * self.prev_steering +
            0.3 * steering
        )

        steering = max(
            -1.0,
            min(1.0, steering)
        )

        self.prev_steering = steering


        # --------------------------------------------------
        # Throttle
        # --------------------------------------------------

        if abs(steering) > 0.40:
            throttle = 0.28

        elif abs(steering) > 0.20:
            throttle = 0.32

        else:
            throttle = 0.38


        # --------------------------------------------------
        # Display
        # --------------------------------------------------

        cv2.putText(
            img_arr,
            f"Steering: {steering:.2f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.putText(
            img_arr,
            f"Throttle: {throttle:.2f}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.line(
            roi,
            (width // 2, 0),
            (width // 2, roi.shape[0]),
            (255, 0, 0),
            2
        )

        # Show lane mask
        img_arr[
            int(height * 0.75):height, :
        ] = cv2.cvtColor(
            mask,
            cv2.COLOR_GRAY2BGR
        )

        return steering, throttle, img_arr
