"""
route3_verify.py  --  Screenshot-diff verification for Route 3 actions.

After a Route 3 action executes, this module:
  1. Waits VERIFY_DELAY_SECONDS for the screen to settle
  2. Captures the post-action screenshot (full primary monitor)
  3. Computes a pixel-level diff against the pre-action screenshot
  4. Returns a verification outcome and the fraction of changed pixels

Verification outcomes:
  "verified"                     Meaningful change detected — action likely worked
  "unverified_no_change"         Little/no change — action may have failed
  "unverified_unexpected_change" Very large change — something unexpected
  "unverified_capture_error"     Screenshot capture failed (non-fatal)

Diff method: per-pixel absolute intensity delta > PIXEL_CHANGE_THRESHOLD
counts as "changed". No additional dependencies beyond mss and cv2, which
are already in the project.

All thresholds are configurable in route3_config.py.
Chosen values (pending tuning from real verification logs):
  VERIFY_CHANGE_THRESHOLD     = 0.02   (2% of pixels)
  VERIFY_UNEXPECTED_THRESHOLD = 0.50  (50% of pixels)
  PIXEL_CHANGE_THRESHOLD      = 10    (per-channel intensity units)
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
    VERIFY_CHANGE_THRESHOLD,
    VERIFY_UNEXPECTED_THRESHOLD,
)

# Per-pixel intensity change to count as "changed".
# Not in route3_config.py — this is an implementation detail, not a tuning knob.
_PIXEL_CHANGE_THRESHOLD = 10


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


def verify_action(pre_screenshot:  np.ndarray | None,
                  delay_seconds:   float = VERIFY_DELAY_SECONDS
                  ) -> tuple[str, float]:
    """
    Wait for the screen to settle, then diff against the pre-action screenshot.

    Args:
      pre_screenshot: BGR numpy array captured BEFORE the action was dispatched
      delay_seconds:  how long to wait before capturing the post-action shot

    Returns:
      (outcome_string, change_fraction)

    This function is called from _run_action in main.py after the dispatch
    completes.  It never raises — failures return "unverified_capture_error".
    """
    if pre_screenshot is None:
        return "unverified_capture_error", 0.0

    try:
        time.sleep(delay_seconds)
        post_screenshot, _ = capture_fullscreen()
        return compute_diff(pre_screenshot, post_screenshot)
    except Exception as e:
        print(f"[Route3Verify] Verify error (non-fatal): {e}")
        return "unverified_capture_error", 0.0
