import time
class Circle():
    def __init__(self,pid=None, cfg = None):
        self.pid=pid
        self.cfg=cfg
    def run(self,img_arr):
        t=time.time()
        if t<10:
            steering = 1.0
            throttle = 0.45
        elif t>=10 and t<=19:
            steering = -1.0
            throttle = 0.45
        else:
            steering = 0.0
            throttle = 0.0
        return steering, throttle, img_arr
    
