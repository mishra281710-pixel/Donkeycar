import cv2
import numpy as np
import time


class LaneFollower:
    """
    STATES:
      LANE_FOLLOW    -> normal lane following. Watches for ONE horizontal
                         yellow element (shape-based: wide-and-short blob,
                         not full-frame coverage - works even if the car
                         isn't perfectly centered).
      DRIVE_PAST     -> once the line is detected, KEEP lane-following
                         normally for exactly `drive_past_time` seconds
                         (positions the car past the parking box).
      REVERSE_TURN   -> reverse while turning RIGHT, toward the box.
      REVERSE_COUNTER -> reverse while turning LEFT (counter-steer), to
                         straighten the car's angle out inside the box.
      REVERSE_STRAIGHTEN -> keep reversing with wheels straight.
      PARK_STOP      -> brief settle/brake.
      PARKED         -> stopped, done - should now be sitting in the box,
                         before (not past) the horizontal line.

    TUNE ON YOUR CAR (this is the important part - there's no distance
    sensor here, so positioning is all timing-based):
      drive_past_time        - how long to keep driving after seeing the
                                line before starting to reverse. Bigger =
                                car ends up further past the line first.
      reverse_turn_time / reverse_steer - how far/hard it swings right first.
      reverse_counter_time / reverse_counter_steer - how far/hard it counter-
                                steers left to straighten the angle.
      reverse_straighten_time - how long it reverses straight after that,
                                to end up parallel inside the box.
      Together, drive_past_time + reverse_turn_time + reverse_counter_time +
      reverse_straighten_time is what determines whether the car ends up
      parked BEFORE the line or overshoots past it - tune by testing and
      adjusting one at a time.

      min_line_width_frac / min_aspect_ratio - shape thresholds for what
      counts as "a horizontal element" vs. the normal lane tape.
    """

    def __init__(self, pid=None, cfg=None):
        self.pid = pid
        self.cfg = cfg

        self.prev_steering = 0.0
        self.stop_detected = False
        self.stop_until = 0
        self.base_throttle = 0.4

        # ---- state machine ----
        self.state = "LANE_FOLLOW"
        self.state_start_time = None

        # ---- horizontal element detection ----
        self.min_line_width_frac = 0.30   # blob width must be >= this fraction of ROI width
        self.min_aspect_ratio = 2.0        # blob width must be >= this many times its height
        self.debounce_frames = 4
        self._above_count = 0
        self._below_count = 0
        self.on_line = False

        # ---- timed maneuver (TUNE THESE) ----
        self.drive_past_time = 7.5          # keep driving for 3s after seeing the line

        self.reverse_turn_time = 1.0
        self.reverse_turn_throttle = -0.70
        self.reverse_steer = 1.0            # turning right into the box
        self.reverse_steer_sign = 1         # flip to -1 if car swings the wrong way

        self.reverse_counter_time = 1.0    # counter-steer left to straighten the angle
        self.reverse_counter_throttle = -0.70
        self.reverse_counter_steer = 1.0    # same magnitude, opposite direction to reverse_steer

        self.reverse_straighten_time = 0.8
        self.reverse_straighten_throttle = -0.25

        self.stop_settle_time = 0.3

    # ============================================================
    def run(self, img_arr):
        if img_arr is None:
            return 0.0, 0.0, img_arr
        if time.time() < self.stop_until:
            return 0.0, 0.0, img_arr
        if self.stop_detected and self.state == "LANE_FOLLOW":
            return 0.0, 0.0, img_arr

        height, width, _ = img_arr.shape

        if self.state == "LANE_FOLLOW":
            return self._lane_follow_and_watch(img_arr, height, width)
        elif self.state == "DRIVE_PAST":
            return self._drive_past(img_arr, height, width)
        elif self.state == "REVERSE_TURN":
            return self._reverse_turn(img_arr)
        elif self.state == "REVERSE_COUNTER":
            return self._reverse_counter(img_arr)
        elif self.state == "REVERSE_STRAIGHTEN":
            return self._reverse_straighten(img_arr)
        elif self.state == "PARK_STOP":
            return self._park_stop(img_arr)
        elif self.state == "PARKED":
            cv2.putText(img_arr, "PARKED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return 0.0, 0.0, img_arr

        return 0.0, 0.0, img_arr

    # ============================================================
    # Core lane-centering logic, reused by both LANE_FOLLOW and DRIVE_PAST.
    # ============================================================
    def _drive_lane(self, img_arr, height, width):
        roi = img_arr[int(height * 0.75):height, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([18, 60, 60]), np.array([45, 255, 255]))

        M = cv2.moments(mask)
        steering = self.prev_steering

        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            error = -(cx - (width // 2))
            steering = -float(error) / (width // 2)
            steering = max(min(steering, 1.0), -1.0)

        steering = 0.7 * self.prev_steering + 0.7 * steering
        self.prev_steering = steering

        if abs(steering) > 0.40:
            throttle = 0.28
        elif abs(steering) > 0.20:
            throttle = 0.32
        else:
            throttle = 0.38

        img_arr[int(height * 0.75):height, :] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return steering, throttle, mask

    # ============================================================
    # STATE: LANE_FOLLOW (priority: stay in lane, also watch for the line)
    # ============================================================
    def _lane_follow_and_watch(self, img_arr, height, width):

        # ---- stop marker (red), unchanged ----
        stop_roi = img_arr[int(height * 0.70):height, :]
        hsv_stop = cv2.cvtColor(stop_roi, cv2.COLOR_RGB2HSV)
        m1 = cv2.inRange(hsv_stop, np.array([0, 120, 70]), np.array([10, 255, 255]))
        m2 = cv2.inRange(hsv_stop, np.array([170, 120, 70]), np.array([180, 255, 255]))
        red_pixels = cv2.countNonZero(cv2.bitwise_or(m1, m2))
        if red_pixels > 100:
            self.stop_until = time.time() + 5
            return 0.0, 0.0, img_arr

        steering, throttle, mask = self._drive_lane(img_arr, height, width)

        # ---- HORIZONTAL ELEMENT WATCH (shape-based) ----
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found_horizontal = False
        best_w = 0
        min_w_px = self.min_line_width_frac * width

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < min_w_px:
                continue
            if h <= 0 or (w / h) < self.min_aspect_ratio:
                continue
            found_horizontal = True
            best_w = max(best_w, w)

        # Uncomment while tuning:
        # print(f"[LANE_FOLLOW] found_horizontal={found_horizontal} best_w={best_w}")

        if found_horizontal:
            self._above_count += 1
            self._below_count = 0
        else:
            self._below_count += 1
            self._above_count = 0

        if self._above_count == self.debounce_frames and not self.on_line:
            self.on_line = True
            print(f"[LANE_FOLLOW] horizontal line detected (width={best_w}px) -> driving past for {self.drive_past_time}s")
            self._enter_state("DRIVE_PAST")
            return steering, throttle, img_arr

        cv2.putText(img_arr, "LANE_FOLLOW", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return steering, throttle, img_arr

    # ============================================================
    # STATE: DRIVE_PAST - keep lane-following normally for a fixed time
    # ============================================================
    def _drive_past(self, img_arr, height, width):
        if self._elapsed() < self.drive_past_time:
            steering, throttle, _ = self._drive_lane(img_arr, height, width)
            cv2.putText(img_arr, f"DRIVE_PAST {self._elapsed():.1f}s", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            return steering, throttle, img_arr

        self._enter_state("REVERSE_TURN")
        return 0.0, 0.0, img_arr

    # ============================================================
    def _enter_state(self, name):
        print(f"[STATE] -> {name}")
        self.state = name
        self.state_start_time = time.time()

    def _elapsed(self):
        return time.time() - self.state_start_time if self.state_start_time else 0.0

    def _reverse_turn(self, img_arr):
        cv2.putText(img_arr, "REVERSE_TURN", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        steering = self.reverse_steer * self.reverse_steer_sign
        if self._elapsed() < self.reverse_turn_time:
            return steering, self.reverse_turn_throttle, img_arr
        self._enter_state("REVERSE_COUNTER")
        return 0.0, 0.0, img_arr

    def _reverse_counter(self, img_arr):
        cv2.putText(img_arr, "REVERSE_COUNTER", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        # opposite direction to reverse_turn - counter-steers left to
        # straighten the car's angle out inside the box
        steering = -self.reverse_counter_steer * self.reverse_steer_sign
        if self._elapsed() < self.reverse_counter_time:
            return steering, self.reverse_counter_throttle, img_arr
        self._enter_state("REVERSE_STRAIGHTEN")
        return 0.0, 0.0, img_arr

    def _reverse_straighten(self, img_arr):
        cv2.putText(img_arr, "REVERSE_STRAIGHTEN", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if self._elapsed() < self.reverse_straighten_time:
            return 0.0, self.reverse_straighten_throttle, img_arr
        self._enter_state("PARK_STOP")
        return 0.0, 0.0, img_arr

    def _park_stop(self, img_arr):
        if self._elapsed() < self.stop_settle_time:
            return 0.0, 0.0, img_arr
        self._enter_state("PARKED")
        return 0.0, 0.0, img_arr

    def reset_to_lane_follow(self):
        self.state = "LANE_FOLLOW"
        self.state_start_time = None
        self.on_line = False
        self._above_count = 0
        self._below_count = 0
        self.stop_detected = False
        self.stop_until = 0
