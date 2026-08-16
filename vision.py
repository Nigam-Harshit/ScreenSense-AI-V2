from ultralytics import YOLO
import mss
import numpy as np
import cv2

model = YOLO("models/best.pt")

def detect_buttons(left, top, width, height):

    with mss.mss() as sct:

        monitor = {
            "left": left,
            "top": top,
            "width": width,
            "height": height
        }

        screenshot = sct.grab(monitor)

        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    results = model(img, conf=0.6)

    detections = []

    for r in results:
        for box in r.boxes:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = model.names[cls_id]

            detections.append((label, conf, x1, y1, x2, y2))

    return detections