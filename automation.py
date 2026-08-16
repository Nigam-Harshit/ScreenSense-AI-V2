import pyautogui
import time
import subprocess
import mss
import cv2
import numpy as np
import win32gui
import win32api
import ctypes
import os
import win32con
import screen_brightness_control as sbc

# Disable the corner fail-safe to prevent macros crashing when the user's mouse is in a screen corner
pyautogui.FAILSAFE = False

is_muted = False
def click_button(action, detections, left, top):

    for label, conf, x1, y1, x2, y2 in detections:

        if label == action and conf > 0.6:

            center_x = left + (x1 + x2) // 2
            center_y = top + (y1 + y2) // 2

            print("Clicking in 2 seconds...")
            time.sleep(2)

            pyautogui.moveTo(center_x, center_y, duration=0.5)
            pyautogui.click()

            print("Action executed.")
            return True

    return False


def focus_window(hwnd):

    try:
        # Pressing ALT momentarily allows us to bypass the Windows Foreground Lock Timeout
        pyautogui.press("alt")
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)

        time.sleep(0.4)

        print("Window focused.")

    except Exception as e:
        print("Could not focus window:", e)


def open_app(app):

    # common system apps
    system_apps = {
        "notepad": "notepad",
        "calculator": "calc",
        "calc": "calc",
        "paint": "mspaint",
        "explorer": "explorer",
        "cmd": "cmd"
    }

    if app in system_apps:
        subprocess.Popen(system_apps[app])
        print("Opening", app)
        return

    print(f"Searching and opening '{app}'...")
    pyautogui.press('win')
    time.sleep(0.5)
    pyautogui.write(app, interval=0.05)
    time.sleep(1)
    pyautogui.press('enter')


def take_screenshot():

    try:
        with mss.mss() as sct:

            monitor = sct.monitors[1]

            screenshot = sct.grab(monitor)

            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            import time
            filename = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, img)

        print(f"Screenshot saved as {filename}")

    except Exception as e:
        print("Screenshot failed:", e)


def move_window(hwnd, position):

    try:
        import win32con
        import pyautogui
        # Required to un-maximize the window before locking to coordinate grid
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        pyautogui.press("alt")
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass

    screen_width = win32api.GetSystemMetrics(0)
    screen_height = win32api.GetSystemMetrics(1)

    if position == "left":
        x = 0
        y = 0
        w = screen_width // 2
        h = screen_height

    elif position == "right":
        x = screen_width // 2
        y = 0
        w = screen_width // 2
        h = screen_height

    elif position == "top":
        x = 0
        y = 0
        w = screen_width
        h = screen_height // 2

    elif position == "bottom":
        x = 0
        y = screen_height // 2
        w = screen_width
        h = screen_height // 2

    elif position == "fullscreen":
        x = 0
        y = 0
        w = screen_width
        h = screen_height

    else:
        print("Invalid move command")
        return

    win32gui.MoveWindow(hwnd, x, y, w, h, True)

    print(f"Window moved to {position}.")


def force_close_process(app):

    try:
        process_name = app + ".exe"

        subprocess.run(
            ["taskkill", "/IM", process_name, "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        print(f"Force closed {process_name}")

    except:
        print("Process close failed.")


def snap_window(position):

    time.sleep(0.3)

    if position == "left":
        pyautogui.hotkey("win", "left")

    elif position == "right":
        pyautogui.hotkey("win", "right")

    elif position == "up":
        pyautogui.hotkey("win", "up")

    elif position == "down":
        pyautogui.hotkey("win", "down")

    time.sleep(0.4)


def increase_brightness():

    current = sbc.get_brightness()[0]
    new = min(current + 10, 100)

    sbc.set_brightness(new)

    print("Brightness:", new)


def decrease_brightness():

    current = sbc.get_brightness()[0]
    new = max(current - 10, 0)

    sbc.set_brightness(new)

    print("Brightness:", new)

def set_brightness(level):
    level = max(0, min(int(level), 100))
    sbc.set_brightness(level)
    print(f"Brightness set to {level}")


def volume_up():

    for _ in range(5):
        pyautogui.press("volumeup")

    print("Volume increased")


def volume_down():

    for _ in range(5):
        pyautogui.press("volumedown")

    print("Volume decreased")

def mute_volume():

    global is_muted

    if not is_muted:
        pyautogui.press("volumemute")
        is_muted = True
        print("Muted")
    else:
        print("Volume already muted")


def unmute_volume():

    global is_muted

    if is_muted:
        pyautogui.press("volumemute")
        is_muted = False
        print("Unmuted")
    else:
        print("Volume already unmuted")

def set_volume(level):

    level = max(0, min(level, 100))

    steps = level // 2

    pyautogui.press("volumemute")
    pyautogui.press("volumeup")

    for _ in range(50):
        pyautogui.press("volumedown")

    for _ in range(steps):
        pyautogui.press("volumeup")

    print(f"Volume set to {level}")
    
def lock_screen():

    ctypes.windll.user32.LockWorkStation()
    print("Screen locked")


def sleep_pc():

    os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
    print("System going to sleep")

def scroll_down():
    pyautogui.scroll(-800)
    print("Scrolled down")

def scroll_up():
    pyautogui.scroll(800)
    print("Scrolled up")

def media_play_pause():
    pyautogui.press("playpause")
    print("Play/Pause media")

def media_next():
    pyautogui.press("nexttrack")
    print("Next track")

def media_prev():
    pyautogui.press("prevtrack")
    print("Previous track")

def new_desktop():
    pyautogui.hotkey("win", "ctrl", "d")
    print("Created new virtual desktop")

def next_desktop():
    pyautogui.hotkey("win", "ctrl", "right")
    print("Switched to next desktop")

def prev_desktop():
    pyautogui.hotkey("win", "ctrl", "left")
    print("Switched to previous desktop")

def close_desktop():
    pyautogui.hotkey("win", "ctrl", "f4")
    print("Closed virtual desktop")

def type_text(text):
    pyautogui.write(text, interval=0.02)
    print(f"Typed text: {text}")