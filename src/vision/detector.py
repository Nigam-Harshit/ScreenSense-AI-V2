import cv2
import numpy as np
import pyautogui

PRIMARY_REGION = (0, 0, 2880, 1800)

def find_template(template_path):
    screenshot = pyautogui.screenshot(region=PRIMARY_REGION)
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

    template = cv2.imread(template_path)

    result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    print("Match confidence:", max_val)

    if max_val > 0.65:   # slightly reduce threshold
        h, w = template.shape[:2]
        center_x = max_loc[0] + w // 2
        center_y = max_loc[1] + h // 2
        return center_x, center_y

    return None
