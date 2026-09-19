"""
vlm.py  --  Route 3: Gemini VLM fallback handler.

Entry point: route3_handle(command, action_hint, confidence, log_entry)

Orchestration:
  1. Capture pre-action screenshot  (route3_verify.capture_fullscreen)
  2. Compute screen perceptual hash (route3_verify._perceptual_hash)
  3. Check semantic cache           (route3_cache.SemanticCache.query)
  4. On cache miss: call Gemini VLM (_call_gemini)
  5. Store response in cache        (route3_cache.SemanticCache.store)
  6. Log invocation to JSONL        (route3_logger.Route3Logger.log_invocation)
  7. Return (action, target, pre_screenshot, call_id)

Post-dispatch verification and verification logging are handled by the
caller (_run_action in main.py) using the returned pre_screenshot and call_id.

Dependencies introduced:
  - google-generativeai  (new; add to requirements.txt)
  - Pillow               (new explicit dep; already present via pyautogui)
  All others (mss, cv2, sentence-transformers) already in the project.

API key: read from GEMINI_API_KEY environment variable or .env file.
If the key is missing, Route 3 logs the failure distinctly (error='gemini_key_missing')
and falls back to dispatching the NLP hint — the session continues, but the console
prints a loud warning so the user knows VLM is NOT running.
"""

import os
import json
import time
import uuid
import numpy as np

from route3_config  import (
    ROUTE3_CACHE_ENABLED,
    VLM_MODEL_NAME,
    VLM_API_KEY_ENV,
    DISRUPTIVE_TRIGGERS,
)
from route3_cache   import get_cache
from route3_logger  import get_route3_logger
from route3_verify  import capture_fullscreen

# ── Valid intent vocabulary (must match nlp.py + dispatch table in main.py) ────
VALID_ACTIONS = [
    "brightness_up", "brightness_down", "set_brightness",
    "volume_up", "volume_down", "set_volume", "mute_volume", "unmute_volume",
    "open_app", "close_button", "minimize_button", "maximize_button",
    "split_apps", "focus_window",
    "move_left", "move_right", "move_top", "move_bottom", "move_fullscreen",
    "screenshot", "lock_screen", "sleep_pc",
    "scroll_down", "scroll_up",
    "media_play_pause", "media_next", "media_prev",
    "new_desktop", "next_desktop", "prev_desktop", "close_desktop",
    "type_text",
    "click_element",
    "unknown",   # returned when command is genuinely unactionable
]

# ── Prompt template ────────────────────────────────────────────────────────────
_PROMPT = """\
You are ScreenSense AI, a Windows desktop voice-command controller.

User said: "{command}"
NLP classifier hint: "{action_hint}" (confidence {confidence:.0%} — below threshold, needs verification)

Look at the attached screenshot of the current screen.
Based on what you see AND what the user said, determine the single best action.

Respond with ONLY valid JSON — no markdown fences, no commentary:
{{
  "action": "<one of the valid actions>",
  "target": "<app name | number | text | null>",
  "target": "<app name | number | text | [x, y] coordinates | null>",
  "reasoning": "<one sentence>",
  "confidence": <0.0 to 1.0>
}}

Valid actions: {valid_actions}

Rules:
- Use "unknown" if the command is unactionable or genuinely unclear even with the screenshot.
- "target" is null for actions that need no argument (lock_screen, scroll_down, etc.).
- "target" is the app name for: open_app, close_button, minimize_button, maximize_button, focus_window.
- "target" is [x, y] screen coordinates (integers) for: click_element.
- If the user says "click on X", "tap X", "press X", or references a visible UI button, contact, chat, name, or link on screen, return action "click_element" with its [x, y] coordinates.
- NEVER default a click request, contact name, or proper noun to "type_text". "type_text" is ONLY for dictating text into an active input field.
- "target" is the text content for: type_text.
- "target" is the numeric level (integer) for: set_volume, set_brightness.
- "target" is [app1, app2] for: split_apps.
"""

# ── API key loading ────────────────────────────────────────────────────────────

def _load_api_key() -> str | None:
    """Try environment variable first, then .env file (no python-dotenv needed)."""
    key = os.environ.get(VLM_API_KEY_ENV)
    if key:
        return key
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_file):
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{VLM_API_KEY_ENV}=") and not line.startswith("#"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def check_disruptive_trigger(action: str | None, raw_command: str) -> tuple[bool, str | None]:
    """Sanity gate: refuse disruptive actions if raw command lacks trigger words."""
    if not action or action not in DISRUPTIVE_TRIGGERS:
        return True, None
    cmd_lower = raw_command.lower()
    triggers = DISRUPTIVE_TRIGGERS[action]
    for trig in triggers:
        if isinstance(trig, (tuple, list)):
            if all(w in cmd_lower for w in trig):
                return True, None
        elif isinstance(trig, str):
            if trig in cmd_lower:
                return True, None
    return False, "disruptive_action_without_trigger"


# ── Gemini VLM call ────────────────────────────────────────────────────────────

def _call_gemini(command:        str,
                 action_hint:    str,
                 confidence:     float,
                 screenshot_bgr: np.ndarray | None
                 ) -> tuple[str | None, object, str, float, float]:
    """
    Call Gemini VLM with the command text, NLP hint, and optional screenshot.

    Returns:
      (action, target, reasoning, vlm_confidence, latency_ms)
    Falls back to (None, None, reason_string, 0.0, latency_ms)
    on any API, dependency, or parse failure — never raises.
    """
    import cv2

    api_key = _load_api_key()
    if not api_key:
        # Return a sentinel tuple that route3_handle can detect and log distinctly.
        return None, None, "gemini_key_missing", None, 0.0

    try:
        import google.generativeai as genai
        import PIL.Image
    except ImportError as e:
        msg = f"dependency_missing: {e}"
        print(f"[VLM] {msg}")
        return None, None, msg, 0.0, 0.0

    genai.configure(api_key=api_key)
    genai.configure(api_key=api_key, transport="rest")
    model = genai.GenerativeModel(VLM_MODEL_NAME)

    prompt = _PROMPT.format(
        command=command,
        action_hint=action_hint,
        confidence=confidence,
        valid_actions=", ".join(VALID_ACTIONS),
    )

    # Build content parts
    parts = [prompt]
    if screenshot_bgr is not None:
        try:
            import cv2
            rgb = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            max_dim = 1024
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            pil_img = PIL.Image.fromarray(rgb)
            parts.append(pil_img)
        except Exception as e:
            print(f"[VLM] Screenshot encoding error (continuing without image): {e}")

    t0 = time.perf_counter()
    try:
        response = model.generate_content(
            parts,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=300,
            ),
        )
        raw_text = response.text.strip()
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        msg = f"gemini_api_error: {e}"
        print(f"[VLM] {msg}")
        return None, None, msg, 0.0, latency_ms

    latency_ms = (time.perf_counter() - t0) * 1000
    return _parse_vlm_response(raw_text, action_hint, latency_ms)


def _parse_vlm_response(raw_text: str, action_hint: str, latency_ms: float = 0.0):
    # Parse JSON — handle markdown fences if Gemini wraps the output
    try:
        clean = raw_text
        clean = raw_text.strip()
        if clean.startswith("```"):
            parts_md = clean.split("```")
            clean = parts_md[1]
            if clean.startswith("json"):
                clean = clean[4:]
        parsed       = json.loads(clean.strip())
        action       = str(parsed.get("action", ""))
        target       = parsed.get("target")
        reasoning    = str(parsed.get("reasoning", ""))
        vlm_conf     = float(parsed.get("confidence", 0.5))

        if action not in VALID_ACTIONS:
            msg = f"unrecognised_action: {action}"
            print(f"[VLM] {msg}")
            return None, None, msg, 0.0, latency_ms

        if action == "unknown":
            return None, None, "vlm_unknown", vlm_conf, latency_ms

        if action == "click_element":
            if isinstance(target, (list, tuple)) and len(target) >= 2:
                target = [int(target[0]), int(target[1])]
            elif isinstance(target, str):
                import re
                nums = re.findall(r"\d+", target)
                if len(nums) >= 2:
                    target = [int(nums[0]), int(nums[1])]

        return action, target, reasoning, vlm_conf, latency_ms

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        msg = f"json_parse_error: {e}"
        print(f"[VLM] {msg}. Raw response (first 200 chars): {raw_text[:200]}")
        return None, None, msg, 0.0, latency_ms


# ── Public Route 3 entry point ─────────────────────────────────────────────────

def route3_handle(command:     str,
                  action_hint: str,
                  confidence:  float,
                  log_entry          # CommandLogger entry (for mark_result)
                  ) -> tuple[str | None, object, object, str]:
    """
    Full Route 3 pipeline — pre-dispatch side.

    Steps:
      1. Generate a unique call_id for this invocation
      2. Capture pre-action screenshot + compute screen hash
      3. Query semantic cache (if ROUTE3_CACHE_ENABLED)
      4. On cache miss: call Gemini VLM
      5. Store response in cache (if enabled, action != 'unknown')
      6. Log invocation entry to route3 JSONL

    Returns:
      (action, target, pre_screenshot_array, call_id)

      - action:             intent string (or None if action == 'unknown')
      - target:             slot value (app name, number, etc.) or None
      - pre_screenshot_array: numpy BGR array for post-dispatch verification
      - call_id:            short UUID string for linking invocation + verification logs

    The caller (_run_action in main.py) must:
      a) Dispatch the returned (action, target) through the normal action table
      b) After dispatch, call route3_log_verification(pre_screenshot, outcome, call_id)
         to complete the log entry with the verification result.

    Failures:
      - Returns (action_hint, None, None, call_id) on soft failures
        (missing key, parse error) so the NLP hint is used as a best-effort action
      - Returns (None, None, None, call_id) if action is 'unknown'
        (caller should mark log_entry as 'vlm_unknown' and return)
    """
    t_start  = time.perf_counter()
    call_id  = uuid.uuid4().hex[:8]   # short 8-char ID
    logger   = get_route3_logger()

    # ── 1. Capture screenshot ─────────────────────────────────────────────────
    pre_ss, screen_hash = capture_fullscreen()

    # ── 2. Cache lookup ───────────────────────────────────────────────────────
    cache_hit   = False
    cached_resp = None
    if ROUTE3_CACHE_ENABLED:
        cache      = get_cache()
        cache_hit, cached_resp = cache.query(command, screen_hash)
        cache_hit, cached_resp = cache.query(command, screen_hash, action_hint=action_hint)

    # ── 3a. Cache hit: reuse cached response ──────────────────────────────────
    if cache_hit and cached_resp is not None:
        action     = cached_resp["action"]
        target     = cached_resp.get("target")
        reasoning  = cached_resp.get("reasoning", "")
        vlm_conf   = cached_resp.get("vlm_confidence", 0.0)
        latency_ms = (time.perf_counter() - t_start) * 1000
        sim        = cached_resp.get("_cache_similarity", 0.0)
        print(f"[VLM] Cache hit (sim={sim:.2f}) → '{action}'")

    # ── 3b. Cache miss: call Gemini ───────────────────────────────────────────
    else:
        action, target, reasoning, vlm_conf, vlm_latency = _call_gemini(
            command, action_hint, confidence, pre_ss
        )
        latency_ms = (time.perf_counter() - t_start) * 1000

        # Detect the key-missing sentinel from _call_gemini
        if reasoning == "gemini_key_missing":
            print(
                "\n" + "!" * 60 + "\n"
                "[Route 3] WARNING: GEMINI_API_KEY is not set.\n"
                "          VLM is NOT running — refusal returned.\n"
                "          Set GEMINI_API_KEY in your .env file to enable Route 3.\n"
                "!" * 60 + "\n"
            )
            # Log the invocation with a distinct error marker and null confidence
            logger.log_invocation(
                call_id        = call_id,
                command        = command,
                action_hint    = action_hint,
                confidence     = confidence,
                screen_hash    = screen_hash,
                cache_hit      = False,
                action_taken   = None,
                target_taken   = None,
                reasoning      = "gemini_key_missing",
                vlm_confidence = None,           # null — not a real VLM response
                latency_ms     = latency_ms,
                error          = "gemini_key_missing",
            )
            return None, "gemini_key_missing", None, call_id

        # Sanity gate on disruptive actions
        if action not in (None, "unknown"):
            allowed, refuse_reason = check_disruptive_trigger(action, command)
            if not allowed:
                print(f"[VLM] Refusing disruptive action '{action}' without required trigger words in command: '{command}'")
                action = None
                target = None
                reasoning = refuse_reason

        # Store in cache only if the action is real, valid, and not a refusal
        if ROUTE3_CACHE_ENABLED and action not in (None, "unknown"):
            get_cache().store(command, screen_hash, {
                "action":         action,
                "target":         target,
                "reasoning":      reasoning,
                "vlm_confidence": vlm_conf,
            }, action=action, target=target)

    # ── 4. Log invocation ─────────────────────────────────────────────────────
    logger.log_invocation(
        call_id        = call_id,
        command        = command,
        action_hint    = action_hint,
        confidence     = confidence,
        screen_hash    = screen_hash,
        cache_hit      = cache_hit,
        action_taken   = action,
        target_taken   = target,
        reasoning      = reasoning,
        vlm_confidence = vlm_conf,
        latency_ms     = latency_ms,
        error          = reasoning if action in (None, "unknown") else None,
    )

    # ── 5. Return to caller ───────────────────────────────────────────────────
    if action in (None, "unknown"):
        print(f"[VLM] No actionable command returned — reason: {reasoning}")
        return None, reasoning, None, call_id

    print(f"[VLM] Route 3 → action='{action}' target='{target}' "
          f"latency={latency_ms:.0f}ms cache_hit={cache_hit}")
    return action, target, pre_ss, call_id


def route3_log_verification(call_id:           str,
                             outcome:           str,
                             change_pct:        float,
                             polls_taken:       int  | None = None,
                             time_to_stable_ms: int  | None = None) -> None:
    """
    Write the post-dispatch verification log entry.
    Called from _run_action in main.py after the action completes.
    Never raises.

    polls_taken       : total poll iterations from verify_action()
    time_to_stable_ms : ms to stable-diff detection, or None on timeout
    """
    try:
        get_route3_logger().log_verification(
            call_id              = call_id,
            verification_outcome = outcome,
            change_pct           = change_pct,
            polls_taken          = polls_taken,
            time_to_stable_ms    = time_to_stable_ms,
        )
    except Exception as e:
        print(f"[VLM] Verification log error (non-fatal): {e}")
