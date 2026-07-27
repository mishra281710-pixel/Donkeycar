import time

class AutoDrive:

    def __init__(self,pid = None, cfg =None):
        self.pid = pid
        self.cfg = cfg

        self.start_time = time.time()

    def run(self, img_arr):

        t = time.time() - self.start_time

        steering = 0.0
        throttle = 0.0

        # Drive forward
        if t < 8:
            steering = 0.0
            throttle = 0.45

        # Stop
        elif t < 9:
            steering = 0.0
            throttle = 0.0

        # Turn around (adjust duration for your car)
        elif t < 11:
            steering = 1.0
            throttle = 0.35

        # Drive back
        elif t < 19:
            steering = 0.0
            throttle = 0.45

        # Stop
        else:
            steering = 0.0
            throttle = 0.0

        return steering, throttle, img_arr
