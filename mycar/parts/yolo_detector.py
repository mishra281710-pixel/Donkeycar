import cv2
import numpy as np
from ultralytics import YOLO

class YOLOYDetector:
    def __init__(self, model_path="/home/mishr/mycar/models/best.onnx", conf=0.5):
        self.model=YOLO(model_path)
        self.conf=conf
        self.class_names=["human","Cone","Safety-barrier","Safety-bollard","Safety-cone","stop_sign"]
        self.last=[]

    def detect(self,image):
        res=self.model.predict(image,conf=self.conf,verbose=False)
        dets=[]
        if not res: return dets,image
        r=res[0]
        out=image.copy()
        if r.boxes is None:
            self.last=[]; return [],out
        for b in r.boxes:
            x1,y1,x2,y2=map(int,b.xyxy[0].tolist())
            c=float(b.conf[0]); cid=int(b.cls[0])
            name=self.class_names[cid] if cid<len(self.class_names) else str(cid)
            det={"class_id":cid,"class_name":name,"confidence":c,"bbox":(x1,y1,x2,y2)}
            dets.append(det)
            cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(out,f"{name} {c:.2f}",(x1,max(20,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        self.last=dets
        return dets,out

    def _has(self,name):
        return any(d["class_name"]==name for d in self.last)
    def has_stop_sign(self):
        return self._has("stop_sign")
    def has_human(self):
        return self._has("human")
    def has_obstacle(self):
        obs={"human","Cone","Safety-barrier","Safety-bollard","Safety-cone","stop_sign"}
        return any(d["class_name"] in obs for d in self.last)
