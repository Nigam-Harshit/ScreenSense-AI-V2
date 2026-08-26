import builtins
import ctypes

_original_print = builtins.print

def custom_print(*args, **kwargs):
    if len(args) == 2 and str(args[0]).lower() == "voice command:":
        _original_print("VOICE COMMAND:", str(args[1]).lower(), **kwargs)
    else:
        upper_args = [str(arg).upper() for arg in args]
        _original_print(*upper_args, **kwargs)

builtins.print = custom_print

ctypes.windll.user32.SetProcessDPIAware()

import win32gui
import time

from nlp import parse_command, CONF_THRESHOLD
from window_manager import find_window, get_window_rect
from vision import detect_buttons
# voice_pipeline is imported lazily inside the voice branch below
# so that text-mode users never load pvporcupine (Task 3 fix).
from command_logger import CommandLogger

_logger = CommandLogger()

from automation import (
    click_button,
    focus_window,
    open_app,
    take_screenshot,
    move_window,
    force_close_process,
    snap_window,
    increase_brightness,
    decrease_brightness,
    set_brightness,
    volume_up,
    volume_down,
    mute_volume,
    unmute_volume,
    lock_screen,
    sleep_pc,
    set_volume,
    scroll_down,
    scroll_up,
    media_play_pause,
    media_next,
    media_prev,
    new_desktop,
    next_desktop,
    prev_desktop,
    close_desktop,
    type_text
)

print("\nScreenSenseAI running...")
print("Commands ready. Type 'exit' to quit.\n")


def run_command(command):

    exit_phrases = ["exit from screensense", "exit screensense", "shut down screensense", "stop screensense", "quit screensense", "close screensense", "exit"]
    if any(phrase in command for phrase in exit_phrases):
        _logger.log_simple(command, None, None, 1.0, "exit", result="ok")
        print("Shutting down ScreenSenseAI.")
        import os
        os._exit(0)

    action, target, confidence = parse_command(command)

    # Determine route label for the log
    if confidence < CONF_THRESHOLD and action is not None:
        route = "vlm_fallback"
    elif action is None:
        route = "regex_fallback"
    else:
        route = "knn"

    with _logger.log_command(command, action, target, confidence, route) as entry:
        _run_action(command, action, target, confidence, route, entry)


def _dispatch_action(action, target, log_entry):
    """All Routes 1 & 2 dispatch logic -- extracted from _run_action.
    Behavior is identical to the previous inline code.
    Do NOT modify this function for Route 3 concerns."""

    if action == "set_volume":
        set_volume(target)
        return

    if action == "set_brightness":
        set_brightness(target)
        return

    if action == "mute_volume":
        mute_volume()
        return

    if action == "unmute_volume":
        unmute_volume()
        return
    
    if action == "lock_screen":
        lock_screen()
        return

    if action == "sleep_pc":
        sleep_pc()
        return

    if action == "scroll_down": scroll_down(); return
    if action == "scroll_up": scroll_up(); return
    if action == "media_play_pause": media_play_pause(); return
    if action == "media_next": media_next(); return
    if action == "media_prev": media_prev(); return
    if action == "new_desktop": new_desktop(); return
    if action == "next_desktop": next_desktop(); return
    if action == "prev_desktop": prev_desktop(); return
    if action == "close_desktop": close_desktop(); return
    
    if action == "type_text":
        type_text(target)
        return

    if action is None:
        print("Invalid command.")
        log_entry.mark_result("not_found")
        return

    if action == "brightness_up":
        increase_brightness()
        return

    if action == "brightness_down":
        decrease_brightness()
        return

    if action == "volume_up":
        volume_up()
        return

    if action == "volume_down":
        volume_down()
        return

    if action == "split_apps":

        app1, app2 = target

        hwnd1, title1 = find_window(app1)
        hwnd2, title2 = find_window(app2)

        if hwnd1 is None or hwnd2 is None:
            print("One or both applications not found.")
            return

        focus_window(hwnd1)
        snap_window("left")

        focus_window(hwnd2)
        snap_window("right")

        print(f"{app1} and {app2} split screen activated.")
        return

    if action == "screenshot":
        take_screenshot()
        return

    if action == "open_app":
        open_app(target)
        return

    hwnd, title = find_window(target)

    if hwnd is None:
        print("Application not found.")
        log_entry.mark_result("not_found")
        return

    if action == "focus_window":
        focus_window(hwnd)
        return

    if action.startswith("move"):

        if action == "move_left":
            move_window(hwnd, "left")

        elif action == "move_right":
            move_window(hwnd, "right")

        elif action == "move_top":
            move_window(hwnd, "top")

        elif action == "move_bottom":
            move_window(hwnd, "bottom")

        elif action == "move_fullscreen":
            move_window(hwnd, "fullscreen")

        return

    print("MATCH FOUND:", title)

    try:
        left, top, width, height = get_window_rect(hwnd)

        if width <= 0 or height <= 0:
            print("Window is minimized or invalid dimension.")
            return

        detections = detect_buttons(left, top, width, height)

        success = click_button(action, detections, left, top)

        if not success:

            if action == "close_button":

                print("YOLO failed → trying WM_CLOSE")

                win32gui.PostMessage(hwnd, 0x0010, 0, 0)

                time.sleep(1)

                if win32gui.IsWindow(hwnd):

                    print("WM_CLOSE failed → force closing process")

                    force_close_process(target)

            else:
                print("Button not detected.")

    except Exception as e:
        print("Error during window interaction:", e)


# ── Route 3: VLM fallback — _run_action orchestration ─────────────────────────

def _run_action(command, action, target, confidence, route, log_entry):
    """Routes the command through the correct handler.
    Routes 1 & 2 go directly to _dispatch_action.
    Route 3 (low-confidence) invokes the Gemini VLM pipeline first,
    then dispatches the VLM-returned action through the same table."""

    _pre_ss  = None   # pre-action screenshot (set only for Route 3)
    _call_id = None   # Route 3 call ID for log correlation

    # ── Route 3 gate ───────────────────────────────────────────────────────────
    if confidence < CONF_THRESHOLD and action is not None:
        try:
            from vlm import route3_handle
            action, target, _pre_ss, _call_id = route3_handle(
                command, action, confidence, log_entry
            )
        except Exception as e:
            print(f"[ROUTER] Route 3 error: {e}")
            log_entry.mark_result("vlm_error")
            return

        if action is None:
            log_entry.mark_result("vlm_unknown")
            return

    # ── Dispatch (Routes 1, 2, and VLM-resolved Route 3) ──────────────────────
    try:
        _dispatch_action(action, target, log_entry)
    finally:
        # Post-action verification (Route 3 only)
        if _pre_ss is not None and _call_id is not None:
            try:
                from route3_config import ROUTE3_VERIFY_ENABLED
                from route3_verify import verify_action
                from vlm import route3_log_verification
                if ROUTE3_VERIFY_ENABLED:
                    outcome, change_pct, polls_taken, time_to_stable_ms = verify_action(_pre_ss)
                    route3_log_verification(_call_id, outcome, change_pct,
                                            polls_taken, time_to_stable_ms)
                    stable_info = (f"{time_to_stable_ms}ms"
                                   if time_to_stable_ms is not None else "timeout")
                    print(f"[Route3Verify] {outcome} ({change_pct:.1%} pixels changed, "
                          f"{polls_taken} polls, stable={stable_info})")
            except Exception as e:
                print(f"[Route3Verify] Non-fatal error: {e}")


mode = input("TYPE OR VOICE? (T/V): ").strip().lower()


if mode == "v":
    # Lazy import: pvporcupine is only evaluated when voice mode is chosen.
    # Text-mode users can run with pvporcupine uninstalled or key invalid.
    from voice_pipeline import voice_loop
    voice_loop(run_command)

else:

    while True:

        command = input("ENTER COMMAND: ").strip().lower()

        run_command(command)