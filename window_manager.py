"""
window_manager.py  --  Window enumeration, target resolution, and recency tracking.
"""

import win32gui
import win32api
import win32con
import win32process
import psutil

# Session recency stack: stores HWNDs in order of most recent interaction (index 0 = most recent)
_SESSION_RECENT_HWNDS: list[int] = []


def record_window_interaction(hwnd: int, title: str = None):
    """
    Record that a window was interacted with or brought to the foreground.
    Places the HWND at the top (index 0) of the session recency stack.
    """
    global _SESSION_RECENT_HWNDS
    if not hwnd or not win32gui.IsWindow(hwnd):
        return
    if hwnd in _SESSION_RECENT_HWNDS:
        _SESSION_RECENT_HWNDS.remove(hwnd)
    _SESSION_RECENT_HWNDS.insert(0, hwnd)


def is_window_on_primary_monitor(hwnd: int) -> bool:
    """
    Check if the given window handle is located on the primary monitor.
    Falls back to True if monitor enumeration fails.
    """
    try:
        hmon = win32api.MonitorFromWindow(hwnd, win32con.MONITOR_DEFAULTTONEAREST)
        info = win32api.GetMonitorInfo(hmon)
        return bool(info.get("Flags", 0) & win32con.MONITORINFOF_PRIMARY)
    except Exception:
        return True


def _get_process_name(hwnd: int) -> str:
    """Return the lowercase executable name of the process owning the window."""
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        if pid:
            return psutil.Process(pid).name().lower()
    except Exception:
        pass
    return ""


def enum_windows():
    """Enumerate all visible windows with non-empty titles."""
    windows = []

    def callback(hwnd, extra):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                windows.append((hwnd, title))

    win32gui.EnumWindows(callback, None)
    return windows


def get_window_rect(hwnd):
    """Return (left, top, width, height) of the given window."""
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top
    return left, top, width, height


def resolve_target_window(target: str = None) -> tuple[int | None, str | None, tuple | None]:
    """
    Enhanced window resolver that binds a target query to a specific HWND.

    Resolution hierarchy:
    1. Active foreground window (if target is None/generic, or matches title/process).
    2. Session recency stack (_SESSION_RECENT_HWNDS): prioritizes recently used windows.
       - Stale recency-stack hit behavior: If an HWND in the recency stack is no longer
         a valid or visible window (e.g. closed or destroyed since last interaction),
         it is immediately purged from _SESSION_RECENT_HWNDS and resolution continues.
    3. Primary-monitor Z-order search: topmost matching window on the primary monitor.
    4. Any-monitor Z-order search: topmost matching window across any connected monitor.

    Returns:
      (hwnd, title, (left, top, width, height)) or (None, None, None)
    """
    global _SESSION_RECENT_HWNDS

    target_clean = str(target).strip().lower() if target is not None else ""

    # Common generic referents for the active window
    is_generic_active = (not target_clean) or (target_clean in {
        "this window", "the window", "current window", "active window",
        "it", "this", "here", "current", "active", "none"
    })

    # 1. Check current foreground window
    fg_hwnd = win32gui.GetForegroundWindow()
    if fg_hwnd and win32gui.IsWindow(fg_hwnd) and win32gui.IsWindowVisible(fg_hwnd):
        fg_title = win32gui.GetWindowText(fg_hwnd)
        fg_pname = _get_process_name(fg_hwnd)
        if is_generic_active or (target_clean in fg_title.lower()) or (target_clean in fg_pname):
            record_window_interaction(fg_hwnd, fg_title)
            return fg_hwnd, fg_title, get_window_rect(fg_hwnd)

    if is_generic_active:
        # If generic/active requested but foreground window is invalid/desktop, check top recency
        for rec_hwnd in list(_SESSION_RECENT_HWNDS):
            if not win32gui.IsWindow(rec_hwnd) or not win32gui.IsWindowVisible(rec_hwnd):
                if rec_hwnd in _SESSION_RECENT_HWNDS:
                    _SESSION_RECENT_HWNDS.remove(rec_hwnd)
                continue
            rec_title = win32gui.GetWindowText(rec_hwnd)
            record_window_interaction(rec_hwnd, rec_title)
            return rec_hwnd, rec_title, get_window_rect(rec_hwnd)

    # 2. Check session recency stack
    # Stale recency-stack hit handling:
    # Iterate over a copy of the stack so we can safely mutate the original stack upon stale hits.
    for rec_hwnd in list(_SESSION_RECENT_HWNDS):
        if not win32gui.IsWindow(rec_hwnd) or not win32gui.IsWindowVisible(rec_hwnd):
            # STALE RECENCY HIT: The window handle has become invalid or closed since it was recorded.
            # Evict it immediately from the session recency stack to prevent ghost window binding.
            if rec_hwnd in _SESSION_RECENT_HWNDS:
                _SESSION_RECENT_HWNDS.remove(rec_hwnd)
            continue

        # Valid recency candidate: verify non-empty title and title or process match
        rec_title = win32gui.GetWindowText(rec_hwnd)
        if not rec_title or not rec_title.strip():
            continue
        rec_pname = _get_process_name(rec_hwnd)
        if is_generic_active or (target_clean in rec_title.lower()) or (target_clean in rec_pname):
            record_window_interaction(rec_hwnd, rec_title)
            return rec_hwnd, rec_title, get_window_rect(rec_hwnd)

    # 3. Z-order enumeration across visible windows
    candidates = []

    def enum_cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if not title or not title.strip():
            return
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0:
            return

        pname = _get_process_name(hwnd)
        if is_generic_active or (target_clean in title.lower()) or (target_clean in pname):
            on_primary = is_window_on_primary_monitor(hwnd)
            candidates.append((hwnd, title, (left, top, width, height), on_primary))

    win32gui.EnumWindows(enum_cb, None)

    if not candidates:
        return None, None, None

    # Prefer candidates on primary monitor first
    primary_cands = [c for c in candidates if c[3]]
    chosen = primary_cands[0] if primary_cands else candidates[0]

    chosen_hwnd, chosen_title, chosen_rect, _ = chosen
    record_window_interaction(chosen_hwnd, chosen_title)
    return chosen_hwnd, chosen_title, chosen_rect


def find_window(target):
    """
    Backward-compatible find_window API.
    Resolves the target window and returns (hwnd, title).
    """
    hwnd, title, _ = resolve_target_window(target)
    return hwnd, title
