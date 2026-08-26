"""
route3_verify.py  --  Screenshot-diff verification for Route 3 actions.

After a Route 3 action executes, this module polls the screen at a fixed
interval until the UI change is stable (two consecutive polls show the same
diff pattern), then returns the verification outcome.

Algorithm
---------
1. Poll every VERIFY_POLL_INTERVAL_SECONDS up to a total of VERIFY_DELAY_SECONDS.
2. On each poll: capture a new screenshot, compute dHash + per-pixel L2 diff
   against the pre-action screenshot.
3. Debounce: a change must be detected *and* stable across two consecutive polls
   before the loop exits early with "verified".
   - Stable = change_pct from poll N and poll N+1 agree within
     _STABLE_DIFF_TOLERANCE (1 pp).  This guards against falsely declaring
     "verified" mid-animation where the UI is still moving.
4. If the timeout is reached without stable change, fall through to the same
   outcome classification logic as before (unverified_no_change,
   unverified_unexpected_change, unverified_capture_error).

Return value of verify_action()
--------------------------------
  (outcome, change_pct, polls_taken, time_to_stable_ms)

  outcome          : str   -- same four strings as before
  change_pct       : float -- fraction of changed pixels in the final diff
  polls_taken      : int   -- total number of poll iterations executed
  time_to_stable_ms: int | None
                           -- ms from action dispatch to stable-diff detection,
                              or None if the loop timed out without stabilising

Verification outcomes (unchanged from Phase 5):
  "verified"                     Meaningful, stable change detected
  "unverified_no_change"         Little/no change — action may have failed
  "unverified_unexpected_change" Very large change — something unexpected
  "unverified_capture_error"     Screenshot capture failed (non-fatal)

Diff method: per-pixel absolute intensity delta > _PIXEL_CHANGE_THRESHOLD
counts as "changed". No additional dependencies beyond mss and cv2.

All tuning constants are in route3_config.py; _PIXEL_CHANGE_THRESHOLD and
_STABLE_DIFF_TOLERANCE are implementation-level constants, not user knobs.
"""

import time
import numpy as np

try:
    import mss
    import cv2
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

from route3_config import (
    VERIFY_DELAY_SECONDS,
    VERIFY_POLL_INTERVAL_SECONDS,
    VERIFY_CHANGE_THRESHOLD,
    VERIFY_UNEXPECTED_THRESHOLD,
)

# Per-pixel intensity delta (per channel) to count as "changed".
# Not a user-facing tuning knob — kept as an implementation constant.
_PIXEL_CHANGE_THRESHOLD = 10

# Two consecutive polls must agree within this many percentage points of
# changed-pixel fraction for the UI to be considered "stable".
# 0.01 = 1 pp tolerance — tight enough to catch real settling, loose enough
# not to spin forever on minor aliasing noise between identical frames.
_STABLE_DIFF_TOLERANCE = 0.01


# ── Screen capture ─────────────────────────────────────────────────────────────

def capture_fullscreen() -> tuple[np.ndarray | None, str]:
    """
    Capture the full primary monitor.

    Returns:
      (bgr_array, perceptual_hash_string)
      (None, "")  if mss/cv2 are not available or capture fails
    """
    if not _DEPS_AVAILABLE:
        return None, ""
    try:
        with mss.mss() as sct:
            monitor = sct.monitors[1]           # index 0 = all monitors combined
            shot    = sct.grab(monitor)
            img     = np.array(shot)
            img     = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img, _perceptual_hash(img)
    except Exception as e:
        print(f"[Route3Verify] Capture error (non-fatal): {e}")
        return None, ""


def _perceptual_hash(img: np.ndarray, grid: int = 16) -> str:
    """
    dHash (difference hash) — fast, no extra library needed.

    Resizes to (grid+1, grid), converts to grayscale, computes horizontal
    gradient, returns the first 32 hex chars of the bit-array.
    """
    small = cv2.resize(img, (grid + 1, grid), interpolation=cv2.INTER_AREA)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    diff  = gray[:, 1:] > gray[:, :-1]          # boolean (grid, grid)
    return diff.flatten().tobytes().hex()[:32]


# ── Diff computation (unchanged from Phase 5) ──────────────────────────────────

def compute_diff(before: np.ndarray | None,
                 after:  np.ndarray | None) -> tuple[str, float]:
    """
    Compute pixel-level change between two full-screen BGR arrays.

    Args:
      before: pre-action screenshot (BGR numpy array)
      after:  post-action screenshot (BGR numpy array)

    Returns:
      (outcome_string, change_fraction)
    """
    if before is None or after is None:
        return "unverified_capture_error", 0.0

    # Ensure same dimensions (e.g. if resolution changed — unlikely but safe)
    if before.shape != after.shape:
        after = cv2.resize(after, (before.shape[1], before.shape[0]),
                           interpolation=cv2.INTER_AREA)

    diff        = cv2.absdiff(before, after)                    # (H, W, 3) uint8
    changed     = np.any(diff > _PIXEL_CHANGE_THRESHOLD, axis=2)  # (H, W) bool
    change_pct  = float(changed.mean())

    if change_pct >= VERIFY_UNEXPECTED_THRESHOLD:
        outcome = "unverified_unexpected_change"
    elif change_pct >= VERIFY_CHANGE_THRESHOLD:
        outcome = "verified"
    else:
        outcome = "unverified_no_change"

    return outcome, change_pct


# ── Adaptive polling verification ──────────────────────────────────────────────

def verify_action(pre_screenshot: np.ndarray | None,
                  ) -> tuple[str, float, int, int | None]:
    """
    Adaptive-polling replacement for the previous fixed-sleep verify_action().

    Polls the screen at VERIFY_POLL_INTERVAL_SECONDS intervals up to
    VERIFY_DELAY_SECONDS (timeout ceiling).  Exits early as soon as a
    meaningful diff is detected *and* is stable across two consecutive polls
    (debounce guard against mid-animation false positives).

    Args:
      pre_screenshot: BGR numpy array captured BEFORE the action was dispatched.
                      Produced by capture_fullscreen() inside route3_handle().

    Returns:
      (outcome, change_pct, polls_taken, time_to_stable_ms)

      outcome           : str       -- same four outcome strings as Phase 5
      change_pct        : float     -- fraction of changed pixels in final diff
      polls_taken       : int       -- total poll iterations executed
      time_to_stable_ms : int|None  -- ms to stable detection, or None on timeout

    Never raises — all failures return "unverified_capture_error".

    Called from _run_action() in main.py inside a try/finally block.
    Callers must unpack all four values (see main.py line 281).
    """
    if pre_screenshot is None:
        return "unverified_capture_error", 0.0, 0, None

    try:
        t_start     = time.perf_counter()
        deadline    = t_start + VERIFY_DELAY_SECONDS

        polls_taken      = 0
        prev_change_pct  = None   # change_pct from the previous poll
        prev_screenshot  = None   # screenshot from the previous poll

        while time.perf_counter() < deadline:
            time.sleep(VERIFY_POLL_INTERVAL_SECONDS)
            polls_taken += 1

            curr_ss, _ = capture_fullscreen()
            if curr_ss is None:
                # Capture failed this poll — treat as no-change, keep polling
                prev_change_pct = None
                prev_screenshot = None
                continue

            _, change_pct = compute_diff(pre_screenshot, curr_ss)

            # ── Debounce check ────────────────────────────────────────────────
            # Require that two consecutive polls both show a meaningful change
            # AND that the two polls agree on the magnitude (stable, not still
            # animating).  "Stable" = change_pct difference within tolerance.
            if (prev_change_pct is not None
                    and change_pct >= VERIFY_CHANGE_THRESHOLD
                    and abs(change_pct - prev_change_pct) <= _STABLE_DIFF_TOLERANCE):

                # Stable meaningful diff detected — exit early
                elapsed_ms        = int((time.perf_counter() - t_start) * 1000)
                outcome, final_pct = compute_diff(pre_screenshot, curr_ss)
                return outcome, final_pct, polls_taken, elapsed_ms

            prev_change_pct = change_pct
            prev_screenshot = curr_ss

        # ── Timeout path (unchanged from Phase 5 logic) ───────────────────────
        # Use the last captured screenshot for the final diff classification.
        # If we never got a successful capture, return a capture error.
        if prev_screenshot is None:
            return "unverified_capture_error", 0.0, polls_taken, None

        outcome, final_pct = compute_diff(pre_screenshot, prev_screenshot)
        return outcome, final_pct, polls_taken, None

    except Exception as e:
        print(f"[Route3Verify] Verify error (non-fatal): {e}")
        return "unverified_capture_error", 0.0, 0, None
