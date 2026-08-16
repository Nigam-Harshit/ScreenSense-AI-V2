import ctypes
ctypes.windll.user32.SetProcessDPIAware()

import win32gui
import mss
import numpy as np
import cv2
import pyautogui
from ultralytics import YOLO
import time

model = YOLO("models/best.pt")

# NEW: Take full command instead of just app name
command = input("Enter command (e.g., close chrome): ").lower()

# NEW: Extract action
if "close" in command:
    action = "close_button"
elif "minimize" in command:
    action = "minimize_button"
elif "maximize" in command:
    action = "maximize_button"
else:
    print("Invalid command.")
    exit()

# NEW: Extract target app (last word)
target = command.split()[-1]

def enum_windows_callback(hwnd, windows):
    if win32gui.IsWindowVisible(hwnd):
        title = win32gui.GetWindowText(hwnd)
        if title:
            windows.append((hwnd, title))

windows = []
win32gui.EnumWindows(enum_windows_callback, windows)

for hwnd, title in windows:
    if target in title.lower():

        print("\nMATCH FOUND:", title)

        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top

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

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                label = model.names[cls_id]

                print(f"Detected: {label} | Confidence: {conf:.2f}")

                # CHANGED: Match detected label with requested action
                if label == action and conf > 0.6:

                    center_x = left + (x1 + x2) // 2
                    center_y = top + (y1 + y2) // 2

                    print("Clicking in 2 seconds...")
                    time.sleep(2)

                    pyautogui.moveTo(center_x, center_y, duration=0.5)
                    pyautogui.click()

                    print(f"Clicked {action}.")
                    break

        break