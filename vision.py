"""
vision.py  --  YOLOv8 UI-element detection cropped to target window.
"""

from ultralytics import YOLO
import mss
import numpy as np
import cv2

model = YOLO("models/best.pt")


def detect_buttons(left, top, width, height):
    """
    Detect UI window control buttons (close_button, minimize_button, maximize_button).
    Constrains the target window bounding box to the primary monitor, crops the capture
    strictly to that region, and maps detected coordinates relative to (left, top).
    """
    with mss.mss() as sct:
        # Resolve primary monitor boundaries
        prim = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        p_left = prim["left"]
        p_top = prim["top"]
        p_right = p_left + prim["width"]
        p_bottom = p_top + prim["height"]

        # Constrain window rect to primary monitor
        w_right = left + width
        w_bottom = top + height

        crop_left = max(p_left, left)
        crop_top = max(p_top, top)
        crop_right = min(p_right, w_right)
        crop_bottom = min(p_bottom, w_bottom)

        crop_width = crop_right - crop_left
        crop_height = crop_bottom - crop_top

        if crop_width <= 0 or crop_height <= 0:
            return []

        monitor = {
            "left": int(crop_left),
            "top": int(crop_top),
            "width": int(crop_width),
            "height": int(crop_height)
        }

        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    results = model(img, conf=0.6)

    detections = []
    # Calculate offset so coordinates align with the caller's (left, top)
    offset_x = crop_left - left
    offset_y = crop_top - top

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Adjust box coordinates to caller window space
            detections.append((
                label,
                conf,
                int(x1 + offset_x),
                int(y1 + offset_y),
                int(x2 + offset_x),
                int(y2 + offset_y)
            ))

    return detections
