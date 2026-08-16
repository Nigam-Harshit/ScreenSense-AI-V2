import pyautogui

pyautogui.FAILSAFE = True

def execute(plan):

    if plan["type"] == "move":
        if plan["direction"] == "right":
            pyautogui.moveRel(100, 0, duration=0.5)
        elif plan["direction"] == "left":
            pyautogui.moveRel(-100, 0, duration=0.5)
        elif plan["direction"] == "up":
            pyautogui.moveRel(0, -100, duration=0.5)
        elif plan["direction"] == "down":
            pyautogui.moveRel(0, 100, duration=0.5)

    elif plan["type"] == "click":
        pyautogui.click()

    elif plan["type"] == "click_at":
        x, y = plan["coords"]
        print("Clicking at:", x, y)
        pyautogui.moveTo(x, y, duration=0.5)
        pyautogui.click(x, y)

    else:
        print("Unknown plan type:", plan["type"])
