"""
routing_config.py  --  Multi-route and dispatch configuration for ScreenSense AI.

Centralises cross-route constants.
Route 1: NLP Direct Dispatch (k-NN >= CONF_THRESHOLD)
Route 2: Vision / YOLO Button Detection
Route 1: Deterministic OS API Calls
Route 2: Fine-Tuned YOLOv8 UI-Element Detection
Route 3: Gemini VLM Fallback
"""

# Intents whose execution is grounded in detectable screen button elements.
# Low-confidence predictions (confidence < CONF_THRESHOLD) for these intents
# will attempt Route 2 (YOLO) detection on the target window before falling
# through to Route 3 (VLM fallback).
# Window control button intents whose execution is strictly grounded in detectable
# screen UI elements via YOLOv8.
# HARD POLICY: These 3 intents must NEVER fall through to Route 3 (VLM) or query/store
# in the semantic cache. If YOLO detection fails on both the primary pass and the single
# cropped retry pass, execution must fail explicitly ("could not find button").
# Zero silent API fallbacks (no WM_CLOSE, no force_close_process).
YOLO_CAPABLE_INTENTS = frozenset({
    "close_button",
    "maximize_button",
    "minimize_button",
})

