# ScreenSense AI — Task Execution Log (Append-Only)

## Session Record: 2026-09-17 (Phases 6A, 7, 8, 9, 10)

### Objectives
- **Phase 6A**: Resolve measurement blockers (Gemini API k
ey verification + `open_app(None)` defensive guard).
- **Phase 7**: Real Route 3 evaluation under live credentials, separate dated telemetry logs, adaptive polling verification, and evaluator comparison against prior baseline.
- **Phase 8**: Route 2 boundary investigation (read-only audit of 32 NLP intents vs 3 YOLO classes, architectural proposals for dispatch gating).
- **Phase 9**: Voice pipeline test (live microphone, openWakeWord model download and detection score, STT latency and routing).
- **Phase 10**: kNN v3 retrain (deferred note).

### Execution Summary & Evidence
1. **Phase 6A-1 (Gemini API Verification)**:
   - Configured `VLM_MODEL_NAME = "gemini-flash-latest"` in `route3_config.py` and `transport="rest"` in `vlm.py`.
   - Verified live HTTP 200 SUCCESS response: `Action: volume_up, Conf: 0.95, Reasoning: 'The user explicitly asked to turn up the sound...'`.
2. **Phase 6A-2 (`open_app(None)` Guard)**:
   - Added guard in `automation.py` lines 56–59 (`if not app: return False`). Tested `automation.open_app(None)` returning `False` cleanly without throwing `TypeError`.
3. **HF Hub Warning Question**:
   - Answered and closed: Model weights are not bundled in repository (`~/.cache` hub cache used). Deployment assumption is (B) fresh download / hub cache check on first run. Standard library behavior, removed from open bug list.
4. **Phase 7 (Route 3 Live Evaluation)**:
   - Executed 24-command test batch under live Gemini key into `logs/commands_2026-09-17.jsonl` and `logs/route3_2026-09-17.jsonl`.
   - Invocations 1–3 captured real Gemini confidence (`0.95`), grounded reasoning, and valid actions.
   - Diagnosed Google Free Tier daily request limit (`20 req/day/project`, `GenerateRequestsPerDayPerProjectPerModel-FreeTier`); verified graceful non-fatal fallback to NLP hint.
   - Evaluated Route 3 adaptive verification debounce: measured `time_to_stable_ms = 801ms` on settled window transition. Diagnosed headless/DPMS monitor standby resulting in 0.00% pixel change in background sessions.
   - Executed `evaluate.py --from 2026-09-17 --to 2026-09-17`: Cache hit rate 4.5% (latency 103.2ms), VLM confidence mean 0.136 (P95 0.950).
5. **Phase 8 (Route 2 Boundary Investigation — Read-Only)**:
   - Audited all 32 NLP intents in `data/intents_v2.json` against all 3 YOLO detection classes (`close_button`, `maximize_button`, `minimize_button`).
   - Documented that 29 out of 32 intents are non-screen-grounded and should not route through YOLO.
   - Proposed Option 1 (Static Route-Capability Mapping Table) as recommended architecture, and Option 2 (Semantic Domain Classifier).
6. **Phase 9 (Voice Pipeline Test)**:
   - Downloaded ONNX model weights (`hey_jarvis_v0.1.onnx`).
   - Verified live microphone capture (`Microphone Array (Intel® Smart Sound)`).
   - Measured wake word score `0.9982` on speech vs `0.0401` ambient (0 false triggers against `OWW_THRESHOLD = 0.5`).
   - Evaluated 7 spoken commands across Routes 1, 2, and 3: 5/7 exact transcriptions, `683.8ms` mean STT latency, `~770ms` total voice-to-dispatch latency.
7. **Phase 10 (kNN v3 Retrain — Deferred)**:
   - v3 retrain remains deferred pending stable v2 baseline measurement; revisit as a separate, isolated experiment once Phase 7/8 data collection is complete.

### Consolidated Deliverable
- Comprehensive report generated: `phase6a-10_report_2026-09-17.md`.

## Session Record: 2026-09-17 (Phase 11: Route 2 Implementation, Headless Capture Fix, Clean Route 3 Measurement, STT Safety)

### Objectives
- **Phase 11A**: Implement Route 2 Option 1 (static mapping table `YOLO_CAPABLE_INTENTS` gating low-confidence button commands through YOLO before Route 3 fallback; bypass non-button commands directly).
- **Phase 11B**: Fix headless/unattended black-frame capture bug in `route3_verify.py` by detecting unrendered frames and recording outcome `"display_unavailable"` instead of misleading `"unverified_no_change"`.
- **Phase 11C**: Execute clean, quota-respecting Route 3 measurement run (<= 15 commands, >= 2 repeats for cache hits, active model with real VLM conf and reasoning, evaluated via `evaluate.py` on separate logs).
- **Phase 11D**: Investigate STT misrecognition safety (check STT confidence availability in `speech_recognition`, propose gating/confirmation design with tradeoffs).

### Execution Summary & Evidence
1. **Phase 11A (Route 2 Option 1 Implementation)**:
   - Defined `YOLO_CAPABLE_INTENTS = frozenset({"close_button", "maximize_button", "minimize_button"})` in `routing_config.py`.
   - Integrated into `main.py` lines 256–289: Low-confidence button commands attempt YOLO before Route 3; non-button commands bypass Route 2 directly to Route 3.
   - Empirical verification: Gate latency 0.002 ms. Tested button commands (`close this window`, `shrink this window`) attempting YOLO; tested non-button commands (`turn up the sound slightly`, `halt audio playback now`) bypassing YOLO with 0 ms vision latency.
2. **Phase 11B (Headless Black-Frame Capture Bug Fix)**:
   - Added `is_valid_capture(img)` in `route3_verify.py` lines 76–87 (checks for None, empty, all zeros, or dynamic range < 2).
   - Integrated into `capture_fullscreen()`, `compute_diff()`, and `verify_action()`. Returns `screen_hash = "display_unavailable"` and `verification_outcome = "display_unavailable"` with 0 polls taken.
   - Evaluator verification: `evaluate.py --self-test` passed all 23 assertions without error.
3. **Phase 11C (Clean Route 3 Measurement Run)**:
   - Clarified that the Phase 7 801 ms verification figure originated from synthetic sequence diffs in `scratch/test_verification_dynamics.py`, not live physical capture.
   - Configured `VLM_MODEL_NAME = "gemini-flash-lite-latest"` in `route3_config.py` (fresh per-model quota bucket).
   - Added image downsampling (max dimension 1024) in `vlm.py` lines 148–156: slashed VLM latency from 43,271 ms to ~2,400 ms (18x speedup).
   - Executed clean 10-command batch into `logs/commands_2026-09-18.jsonl` and `logs/route3_2026-09-18.jsonl`.
   - Results: 0 quota errors (`quota_hit: False`), median VLM confidence = 0.950 (mean = 0.739), cache hit latency = 88.1 ms.
   - Evaluated via `evaluate.py --from 2026-09-18 --to 2026-09-18`.
4. **Phase 11D (STT Misrecognition Safety Investigation)**:
   - Tested Google Web Speech endpoint via `speech_recognition`; confirmed numeric confidence is not returned.
   - Formulated 3-tier safety architecture: multi-candidate intent consensus (`show_all=True`), lexical out-of-domain filtering, and verbal confirmation gating for high-impact intents (`force_close_process`, `lock_screen`, `sleep_pc`).

### Consolidated Deliverable
- Comprehensive report generated: `phase11_report_2026-09-17.md`.

## Session Record: 2026-09-18 (Dry-Run Failure Diagnosis & Fix Brief: Tickets 1–9)

### Execution Summary & Evidence

1. **Ticket 1 (Logging Completeness & Telemetry Instrumentation)**:
   - Extended `_LogEntry` in `command_logger.py` to record `target_hwnd` (`int | None`), `target_title` (`str | None`), and `action_taken` (`str | None`).
   - Added setter methods `entry.set_window(hwnd, title)` and `entry.set_action_taken(action)`.
   - Updated `_LogEntry.to_dict()` and `CommandLogger.log_simple()` with the new telemetry fields.
   - Added `CommandLogger.log_wakeword()` and `CommandLogger.read_wakeword_file()` to write structured wake-word scores to `logs/wakeword_YYYY-MM-DD.jsonl` with ISO-8601 UTC timestamps, detection scores, `OWW_THRESHOLD` sourced directly from `voice_config.py`, RMS audio energy, and status tags (`"triggered"`, `"rejected"`, `"silence"`).
   - Updated `voice_pipeline.py` to log wake-word telemetry on each evaluated chunk and log explicit `route="asr_failure"`, `result="asr_failure"` command entries whenever STT yields no speech.
   - Updated `main.py` to capture empty/blank commands with structured logging and record `target_hwnd`, `target_title`, and `action_taken` across dispatch paths.
   - Verified via `scratch/test_ticket1.py`: `_LogEntry` initialization and setters passed, wake-word telemetry logging passed, empty command logging passed, non-existent window logging passed.
   - Evaluator verification: `evaluate.py --self-test` passed all 23 assertions without error.
2. **Ticket 2 (`OPEN_APP` Fast-Path & Focus-If-Already-Running)**:
   - Added regex fast-path `^(?:open|launch|start)\s+([a-zA-Z0-9_\-\. ]+)$` in `main.py` before `parse_command()`.
   - On match: assigns `action="open_app"`, `target=app_name`, `confidence=1.0`, `route="fast_path"`. Bypasses k-NN classifier and Route 3 VLM fallback entirely.
   - On non-match: explicitly documented in code to fall through cleanly to standard k-NN NLP parsing (`parse_command()`).
   - Updated `open_app()` in `automation.py` to check for an existing open window via `window_manager` before launching a new process; if found, brings existing window to foreground via `focus_window()` and returns `"focused_existing"`, preventing duplicate instances.
   - Hardened `focus_window()` in `automation.py` to handle Windows Foreground Lock restrictions (`SW_RESTORE`, `SW_SHOW`, `BringWindowToTop`).
   - Verified via `scratch/test_ticket2.py`: initial "open notepad" dispatched in 0.151s (< 1.0s) with 0 VLM calls; second "open notepad" focused existing window in 0.519s with `result="focused_existing"`, maintaining single instance; regex non-match ("volume up") cleanly fell through to standard k-NN NLP (`route="knn"`, `action="volume_up"`).
   - Evaluator verification: `evaluate.py --self-test` passed all 23 assertions without error.
3. **Ticket 3 (Window Target Resolution, Recency & Cropped Vision Capture)**:
   - Implemented `resolve_target_window(target)` in `window_manager.py`: prioritizes active foreground window, session recency stack (`_SESSION_RECENT_HWNDS`), and primary-monitor Z-order search matching title and process executable name via `psutil`.
   - Explicitly handled stale recency-stack hits: if a recorded HWND is closed or invalid (`not IsWindow` or `not IsWindowVisible`), it is immediately purged from `_SESSION_RECENT_HWNDS`, falling back cleanly to valid desktop windows.
   - Added `is_window_on_primary_monitor(hwnd)` and `record_window_interaction(hwnd, title)` to maintain session focus history.
   - Updated `detect_buttons()` in `vision.py` to constrain target window bounding boxes to the primary monitor, crop the capture strictly to that region before running YOLO, and offset detected bounding boxes to align with caller coordinate space.
   - Updated `capture_fullscreen()` in `route3_verify.py` to safely index `sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]`.
   - Verified via `scratch/test_ticket3.py`: primary monitor check verified, multi-window disambiguation via recency stack verified, stale HWND purge verified, cropped YOLO inference executed cleanly without error.
   - Evaluator verification: `evaluate.py --self-test` passed all 23 assertions without error.
4. **Ticket 4 (Hard Route-2-Only Policy for Window Controls)**:
   - Updated `routing_config.py` with hard policy documentation: window controls (`close_button`, `minimize_button`, `maximize_button`) must never route to Route 3 (VLM) or query/store in the semantic cache.
   - Enforced Route-2-only dispatch in `main.py::_run_action()` for all three window control intents: resolves target window via `resolve_target_window()`, performs two passes of cropped YOLO button detection, and on detection miss fails explicitly with `[Route 2] Could not find the {action} on '{title}'` and `result="button_not_found"`.
   - Stripped all legacy `WM_CLOSE` and `force_close_process` fallbacks from `_dispatch_action()` and `_run_action()`, bringing `close_button` to exact parity with `minimize_button` and `maximize_button`.
   - Updated `nlp.py` to preserve extracted slot `target` even for low-confidence predictions so window targets reach Route 2 vision.
   - Verified via `scratch/test_ticket4.py`: forced YOLO miss on `close notepad`, `minimize notepad`, and `maximize notepad` reported explicit failure, kept windows open (zero `WM_CLOSE`/kill fallback), produced 0 Route 3 VLM calls, and logged `route="vision"`, `result="button_not_found"`.
   - Evaluator verification: `evaluate.py --self-test` passed all 23 assertions without error.
5. **Ticket 5 (Semantic Cache Intent & Target Binding)**:
   - Redesigned `_cache_key_text()` in `route3_cache.py` to bind intent, target, command, and screen hash: `f"intent:{action} | target:{target} | cmd:{command} [screen:{screen_hash}]"`.
   - Updated `SemanticCache.query()` to accept `action_hint` and `target_hint`, enforcing that candidate cache hits must match `action_hint` (rejecting mismatched actions even if visual similarity meets or exceeds threshold).
   - Enforced hard policy in `route3_cache.py`: window control button intents (`close_button`, `minimize_button`, `maximize_button`) bypass cache lookup and cache storage completely.
   - Updated `vlm.py` to pass `action_hint` to `cache.query()` and `action`/`target` to `cache.store()`.
   - Verified via `scratch/test_ticket5.py`: key formatting validated, intent mismatch rejected on identical screen state, intent match hit verified, window control cache bypass verified.
   - Evaluator verification: `evaluate.py --self-test` passed all 23 assertions without error.
6. **Ticket 6 (Deterministic Window Move & Repositioning)**:
   - Implemented directional-stripping slot extractor `_extract_move_target()` in `nlp.py`: strips directional tokens and noise words (`left`, `right`, `top`, `bottom`, `to`, `the`, `window`, etc.) and extracts the actual application name (`"notepad"`, `"chrome"`, or `None` for active window).
   - Enhanced `_keyword_override()` in `nlp.py` to deterministically disambiguate directional move commands (`move`, `snap`, `shift` + directions).
   - Enforced hard Route 1 policy in `main.py`: `move_*` intents never route to Route 3 VLM fallback.
   - Refined `window_manager.py` to filter out empty-title background helper/tooltip windows during target resolution, binding cleanly to top-level visible application windows.
   - Verified via `scratch/test_ticket6.py`: slot extraction verified across move variants, `move notepad to the left` executed via Win32 API repositioning window to (0, 0) with zero VLM calls, logged with `route="knn"`, `result="ok"`, `target="notepad"`, `target_hwnd` bound.
   - Evaluator verification: `evaluate.py --self-test` passed all 23 assertions without error.

7. **Ticket 7 ("click on <named element>" Intent Taxonomy & VLM Action Selection)**:
   - Added `click_element` intent block to `data/intents_v3.json` with 40 representative natural language examples (e.g., "click on harshit nigam", "tap on the submit button", "select contact", "click save changes").
   - Added `click_element` to `VALID_ACTIONS` in `vlm.py` and enhanced `_PROMPT` rules instructing the model to map references to buttons, contacts, chats, links, or coordinates to `click_element` with `[x, y]` screen coordinates, and never default click requests to `type_text`.
   - Added coordinate normalization in `vlm.py` (`_parse_vlm_response`) to extract and validate `[x, y]` integers from either list or string target representations.
   - Added `click_coordinate(x, y)` in `automation.py` using `pyautogui.moveTo` and `pyautogui.click` with safe error handling.
   - Updated `_dispatch_action()` in `main.py` to handle `click_element`: invokes `click_coordinate()` when valid coordinates are supplied, marks result `"ok"`, and never routes to `type_text`.
   - Verified via `scratch/test_ticket7.py`: `intents_v3.json` definition verified, `vlm.py` actions and prompt rules verified, coordinate normalization verified, `automation.click_coordinate` verified, and `main.py` dispatch verified (routes to coordinate click, never to `type_text`).
   - Evaluator verification: `evaluate.py --self-test` passed all 23 assertions without error.

8. **Ticket 8 (Wake-Word Telemetry Final Audit & Test)**:
   - Audited wake-word telemetry logging across `voice_pipeline.py` and `command_logger.py`: confirmed wake-word threshold `OWW_THRESHOLD` is dynamically imported from `voice_config.py` (value 0.5) and not hardcoded.
   - Verified structured wake-word score records in `CommandLogger.log_wakeword()` writing to `logs/wakeword_YYYY-MM-DD.jsonl` with ISO-8601 UTC millisecond timestamps, model score, threshold, audio RMS energy, and status categorization (`"triggered"`, `"rejected"`, `"silence"`).
   - Confirmed ASR failure logging in `voice_pipeline.py` and `command_logger.py` logs explicit `route="asr_failure"`, `result="asr_failure"`, and descriptive error message whenever STT yields no transcribed speech.
   - Verified via `scratch/test_ticket8.py`: dynamic threshold verification passed, direct logger writes and JSON format passed, simulated stream chunk categorization across all 3 status branches verified, and ASR failure logging verified.
   - Evaluator verification: `evaluate.py --self-test` passed all 23 assertions without error.

9. **Ticket 9 (Consolidated Verification Pass)**:
   - Built and executed `scratch/verify_tickets_1_to_8.py` to systematically validate all architectural fixes and edge cases across Tickets 1 through 8 in an end-to-end dry-run sequence:
     1. Scenario 1 (Fast-Path Launch): `open notepad` executed in 0.138s (< 1.0s target) with 0 VLM fallback calls, logging `route="fast_path"`, `confidence=1.0`, and launching Notepad with HWND bound.
     2. Scenario 2 (Focus Existing): Second `open notepad` detected existing open window in 0.569s, brought it to foreground via `focus_window()`, logged `result="focused_existing"`, and maintained single process instance with zero duplicate processes spawned.
     3. Scenario 3 (Deterministic Move): `move notepad to the left` cleanly resolved target window HWND, repositioned window to (0, 0, 1440, 1800), logged `route="knn"`, `action="move_left"`, and `target_hwnd` bound with 0 VLM calls.
     4. Scenario 4 (Window Controls Hard Route 2 Policy): Forced YOLO miss on `close_button` produced explicit failure (`[Route 2] Could not find the close_button on 'Untitled - Notepad'`, `result="button_not_found"`), 0 VLM fallback calls, and 0 silent kill/WM_CLOSE fallbacks, verifying window remained open and intact.
     5. Scenario 5 (Semantic Cache Intent Binding): Proved that identical screen hash with conflicting action hint is rejected by the cache, matching action hint hits cache, and window control button intents bypass cache completely.
     6. Scenario 6 (`click_element` Dispatch): Verified `click_element` target coordinates `[750, 600]` are dispatched directly to `automation.click_coordinate()`, logged as `result="ok"`, and never routed to `type_text`.
     7. Scenario 7 (Wake-Word Telemetry): Verified wake-word log records exist with ISO-8601 UTC timestamps, dynamic threshold `OWW_THRESHOLD=0.5` from `voice_config.py`, audio RMS energy, and valid status categorization.
     8. Desktop Cleanup: Cleanly terminated all test instances leaving the desktop in an intact state.
   - Evaluation Harness Execution:
     - Ran `python evaluate.py --self-test`: all 23 synthetic data assertions passed.
     - Ran `python evaluate.py --from 2026-09-19 --to 2026-09-19`: measured 0.0% VLM fallback rate for deterministic operations, fast-path mean duration of 390.6ms (P95 638.3ms), k-NN duration of 177.1ms, with 0 unhandled errors.

## Session Record: 2026-09-19 (Brief A — Deterministic Rule Layer for Site & Scroll Commands)

### Objectives
- Implement deterministic rule layer in `site_rules.py` intercepting site aliases, scroll commands with quantity words, and unsupported on-screen content actions before classifier evaluation.
- Extend `automation.py` with `open_website(url)` and add optional `amount=800` parameter to `scroll_down()` and `scroll_up()`.
- Hook `match_rule()` into `main.py::run_command()` with dedicated execution helper `_run_rule_action()` logging via `CommandLogger` with `confidence=1.0`.
- Verify all Appendix Part 1 and Part 2 command rows via standard library `unittest` suite in `test_site_rules.py`.




### Execution Summary & Evidence
1. **Pre-flight (Step 0 Audit)**:
   - Scanned production dataset `data/intents_v2.json` (800 examples across 32 intents).
   - Zero examples in `data/intents_v2.json` start with "click" or "tap".
   - Verified that zero existing labelled examples in `data/intents_v2.json` are intercepted by the new rule layer (clean fall-through preserved for all existing intents).
2. **Deterministic Rule Layer (`site_rules.py`)**:
   - Implemented `match_rule(command: str)` with text normalization (typographic apostrophes replaced, punctuation removed, whitespace collapsed).
   - Rule 1 (Open Website): matches leading open/visit/click verbs with optional trailing site nouns against `SITE_ALIASES` dictionary (`youtube`, `google`, `gmail`, `apple`, `github`, `linkedin`, `stackoverflow`, `kaggle`, `huggingface`).
   - Rule 2 (Scroll): matches `scroll up` / `scroll down` with quantity words (`bit`/`little`/`slightly`/`tiny` -> 300 px; `lot`/`far`/`much` -> 1600 px; labeled "chosen, pending tuning"). Plain scroll commands return `None` to preserve classifier path.
   - Rule 3 (Unsupported Guard): safely intercepts on-screen clicking and ordinal-on-content phrasings without exception words (`close`, `minimize`, `maximize`, `button`, `window`, etc.), returning `("unsupported", ...)` with zero action.
3. **Automation Layer (`automation.py`)**:
   - Added `open_website(url)` invoking `webbrowser.open(url)`.
   - Updated `scroll_down(amount=800)` and `scroll_up(amount=800)` with default `800` parameter preserving backward compatibility.
4. **Main Router Hook (`main.py`)**:
   - Added `match_rule(command)` evaluation in `run_command()` immediately after exit phrase checks and before NLP parsing.
   - Implemented `_run_rule_action(rule_match, log_entry)` helper calling `automation.py` functions directly, bypassing `_dispatch_action`.
   - Logged with routes `rule_site`, `rule_scroll`, and `rule_unsupported` with `confidence=1.0` and results `ok` or `unsupported`.
5. **Unit Testing (`test_site_rules.py`)**:
   - Built comprehensive `unittest` test suite covering all 22 Part 1 rows and all 13 Part 2 rows, plus mocked execution checks for browser opening and scroll amounts.
   - Ran `python test_site_rules.py`: 40/40 tests passed in 0.009s.
   - Verified `evaluate.py --self-test`: 23/23 assertions passed.
   - Live browser smoke test in text mode: **NOT RUN** (as per offline unit test specification).
