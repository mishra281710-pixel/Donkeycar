import cv2
import numpy as np
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite

# ----------------------------
# Load model
# ----------------------------
MODEL_PATH = "detect.tflite"
LABEL_PATH = "labelmap.txt"

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]
floating_model = input_details[0]['dtype'] == np.float32

# ----------------------------
# Load labels
# ----------------------------
with open(LABEL_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

print("Labels loaded:", len(labels))

# ----------------------------
# Camera
# ----------------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

print("Camera Started")

while True:

    frame = picam2.capture_array()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(rgb, (input_width, input_height))
    input_data = np.expand_dims(image, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    detected = False

    for i in range(len(scores)):

        if scores[i] > 0.60:

            class_id = int(classes[i])

            if class_id < len(labels):

                label = labels[class_id]

                print(f"{label} : {scores[i]:.2f}")

                if label.lower() == "stop sign":

                    detected = True

    if detected:
        print("############################")
        print("### STOP SIGN DETECTED ####")
        print("############################")
