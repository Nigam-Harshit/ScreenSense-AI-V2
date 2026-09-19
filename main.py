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

import re
import win32gui
import time

from nlp import parse_command, CONF_THRESHOLD
from routing_config import YOLO_CAPABLE_INTENTS
from window_manager import find_window, get_window_rect
from window_manager import find_window, get_window_rect, resolve_target_window
from vision import detect_buttons
# voice_pipeline is imported lazily inside the voice branch below
# so that text-mode users never load pvporcupine (Task 3 fix).
# so that text-mode users run without loading audio/wake-word dependencies.
from command_logger import CommandLogger

_logger = CommandLogger()

from automation import (
    click_button,
    focus_window,
    open_app,
    take_screenshot,
    move_window,
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
    type_text,
    click_coordinate,
    open_website
)
from site_rules import match_rule

print("\nScreenSenseAI running...")
print("Commands ready. Type 'exit' to quit.\n")


def run_command(command):
    if not command or not command.strip():
        _logger.log_simple(
            raw_command=command or "",
            action=None,
            target=None,
            confidence=0.0,
            route="empty_command",
            result="empty_command",
            error_msg="Empty or blank command received",
        )
        return

    exit_phrases = ["exit from screensense", "exit screensense", "shut down screensense", "stop screensense", "quit screensense", "close screensense", "exit"]
    if any(phrase in command.lower() for phrase in exit_phrases):
        _logger.log_simple(command, None, None, 1.0, "exit", result="ok", action_taken="exit")
        print("Shutting down ScreenSenseAI.")
        import os
        os._exit(0)

    # ── Rule layer for site, scroll, and unsupported commands ─────────────────
    rule_match = match_rule(command)
    if rule_match:
        rule_type = rule_match[0]
        if rule_type == "open_website":
            route = "rule_site"
            target = rule_match[1]
        elif rule_type == "scroll":
            route = "rule_scroll"
            target = (rule_match[1], rule_match[2])
        elif rule_type == "unsupported":
            route = "rule_unsupported"
            target = rule_match[1]
        else:
            route = "rule"
            target = None

        with _logger.log_command(command, rule_type, target, 1.0, route) as entry:
            _run_rule_action(rule_match, entry)
        return

    # ── Ticket 2 Fast-Path: open/launch/start app ─────────────────────────────
    fast_match = re.match(r"^(?:open|launch|start)\s+([a-zA-Z0-9_\-\. ]+)$", command.strip(), re.IGNORECASE)
    if fast_match:
        app_name = fast_match.group(1).strip()
        # Fast-path match: classifies as open_app with confidence 1.0 (Route 1).
        # Bypasses k-NN classifier and Route 3 VLM fallback entirely.
        #
        # Non-match behavior: Any command that does NOT match this regex pattern
        # falls through cleanly below to standard NLP intent classification (parse_command).
        action = "open_app"
        target = app_name
        confidence = 1.0
        route = "fast_path"
    else:
        # Regex non-match: cleanly fall through to standard k-NN NLP parsing
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


def _run_rule_action(rule_match, log_entry):
    """Executes actions matched by the deterministic rule layer."""
    rule_type = rule_match[0]
    if rule_type == "open_website":
        url = rule_match[1]
        open_website(url)
        if log_entry:
            log_entry.set_action_taken("open_website")
            log_entry.mark_result("ok")
    elif rule_type == "scroll":
        direction, amount = rule_match[1], rule_match[2]
        if direction == "down":
            scroll_down(amount)
        elif direction == "up":
            scroll_up(amount)
        if log_entry:
            log_entry.set_action_taken(f"scroll_{direction}")
            log_entry.mark_result("ok")
    elif rule_type == "unsupported":
        reason = rule_match[1]
        print(reason)
        if log_entry:
            log_entry.set_action_taken("unsupported")
            log_entry.mark_result("unsupported", error_msg=reason)


def _dispatch_action(action, target, log_entry):
    """All Routes 1 & 2 dispatch logic -- extracted from _run_action.
    Behavior is identical to the previous inline code.
    Do NOT modify this function for Route 3 concerns."""
    if log_entry:
        log_entry.set_action_taken(action)

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

    if action == "click_element":
        if isinstance(target, (list, tuple)) and len(target) >= 2:
            click_coordinate(target[0], target[1])
            if log_entry:
                log_entry.mark_result("ok")
        else:
            print(f"[Dispatch] Invalid coordinate target for click_element: {target}")
            if log_entry:
                log_entry.mark_result("invalid_target")
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
        result = open_app(target)
        if result == "focused_existing" and log_entry:
            log_entry.mark_result("focused_existing")
        return

    hwnd, title = find_window(target)
    if log_entry:
        log_entry.set_window(hwnd, title)

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
            print(f"[Route 2] Could not find the {action} on '{title}'")
            if log_entry:
                log_entry.mark_result("button_not_found")
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
    # ── Low-confidence handling: Route 2 (YOLO) gate -> Route 3 (VLM) fallback
    # ── Hard Policy: Window control button intents are strictly Route 2 (YOLO) ──
    # Window controls (close_button, minimize_button, maximize_button) must NEVER
    # route to Route 3 (VLM) or semantic cache, and must NEVER have silent API fallbacks
    # (no WM_CLOSE, no process termination). All three button intents follow this exact policy.
    if action in YOLO_CAPABLE_INTENTS:
        if log_entry:
            log_entry.route = "vision"
            log_entry.set_action_taken(action)

        hwnd, title, rect = resolve_target_window(target)
        if log_entry:
            log_entry.set_window(hwnd, title)

        if not hwnd or not win32gui.IsWindow(hwnd):
            print(f"[Route 2] Application '{target}' not found.")
            if log_entry:
                log_entry.mark_result("not_found")
            return

        left, top, width, height = rect if rect else get_window_rect(hwnd)
        if width <= 0 or height <= 0:
            print(f"[Route 2] Window '{title}' has invalid dimensions or is minimized.")
            if log_entry:
                log_entry.mark_result("invalid_dimension")
            return

        # Pass 1: YOLO detection on cropped window region
        detections = detect_buttons(left, top, width, height)
        matched = any(label == action and c >= 0.60 for label, c, *coords in detections)

        # Pass 2: If missed on pass 1, refresh rect and retry once
        if not matched:
            time.sleep(0.2)
            left, top, width, height = get_window_rect(hwnd)
            if width > 0 and height > 0:
                detections = detect_buttons(left, top, width, height)
                matched = any(label == action and c >= 0.60 for label, c, *coords in detections)

        if matched:
            print(f"[Route 2] Button '{action}' detected by YOLO (conf >= 0.60) on '{title}' — executing click")
            success = click_button(action, detections, left, top)
            if success:
                if log_entry:
                    log_entry.route = "vision"
                    log_entry.mark_result("ok")
                return

        # Explicit failure on detection miss for ALL three button intents
        # (close_button matches minimize_button and maximize_button exactly;
        #  zero WM_CLOSE, zero force termination, zero Route 3 VLM fallback)
        print(f"[Route 2] Could not find the {action} on '{title}'")
        if log_entry:
            log_entry.route = "vision"
            log_entry.mark_result("button_not_found")
        return

    # ── Hard Policy: Move/reposition intents are strictly Route 1 (deterministic) ──
    # Moving and snapping windows must NEVER route to Route 3 VLM fallback.
    if action and action.startswith("move_"):
        if log_entry:
            log_entry.route = "knn"
        try:
            _dispatch_action(action, target, log_entry)
        except Exception as e:
            print(f"[Route 1] Move window error: {e}")
            if log_entry:
                log_entry.mark_result("error", str(e))
        return

    # ── Route 3 gate (for non-button low-confidence commands) ─────────────────
    if confidence < CONF_THRESHOLD and action is not None:
        print(f"[ROUTER] Intent '{action}' is not screen-grounded — bypassing Route 2 directly to Route 3")

        # Fall through to Route 3 (VLM fallback)
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
            print("Could not understand that command.")
            reason = target if isinstance(target, str) else "unknown"
            if log_entry:
                log_entry.mark_result("route3_failed", error_msg=reason)
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


if __name__ == "__main__":

    mode = input("TYPE OR VOICE? (T/V): ").strip().lower()

    if mode == "v":
        # Lazy import: voice pipeline is only evaluated when voice mode is chosen.
        # Text-mode users run without loading audio/wake-word dependencies.
        from voice_pipeline import voice_loop
        voice_loop(run_command)

    else:

        while True:

            command = input("ENTER COMMAND: ").strip().lower()

            run_command(command)