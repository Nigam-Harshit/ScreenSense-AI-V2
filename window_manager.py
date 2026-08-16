import win32gui

def enum_windows():

    windows = []

    def callback(hwnd, extra):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                windows.append((hwnd, title))

    win32gui.EnumWindows(callback, None)

    return windows


def find_window(target):

    if str(target).strip() == "" or target is None:
        return None, None

    windows = enum_windows()

    for hwnd, title in windows:
        if str(target).lower() in title.lower():
            return hwnd, title

    return None, None


def get_window_rect(hwnd):

    left, top, right, bottom = win32gui.GetWindowRect(hwnd)

    width = right - left
    height = bottom - top

    return left, top, width, height