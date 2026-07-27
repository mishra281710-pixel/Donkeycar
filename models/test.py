import cv2
import numpy as np
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite

MODEL='detect.tflite'
LABELS='labelmap.txt'
interpreter=tflite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()
inp=interpreter.get_input_details(); out=interpreter.get_output_details()
h,w=inp[0]['shape'][1],inp[0]['shape'][2]
floating=inp[0]['dtype']==np.float32
labels=[l.strip() for l in open(LABELS)]
print('Loaded',len(labels),'labels')
picam2=Picamera2(); picam2.configure(picam2.create_preview_configuration(main={'size':(640,480)})); picam2.start(); print('Camera Started')
while True:
 frame=picam2.capture_array(); frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR); rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); img=cv2.resize(rgb,(w,h)); x=np.expand_dims(img,0)
 if floating: x=(np.float32(x)-127.5)/127.5
 interpreter.set_tensor(inp[0]['index'],x); interpreter.invoke(); boxes=interpreter.get_tensor(out[0]['index'])[0]; classes=interpreter.get_tensor(out[1]['index'])[0]; scores=interpreter.get_tensor(out[2]['index'])[0]; num=int(interpreter.get_tensor(out[3]['index'])[0]); stop=False; print('-'*40)
 for i in range(num):
  sc=float(scores[i])
  if sc<0.2: continue
  cid=int(classes[i]); label=labels[cid] if cid<len(labels) else str(cid); print(f'{i}: {label} ({cid}) {sc:.2f}')
  if label.lower()=='stop sign': stop=True
 if stop: print('######## STOP SIGN DETECTED ########')
