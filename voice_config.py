"""
voice_config.py  --  Centralised config for voice input (wake word + STT).

All voice-related constants live here.  Do NOT scatter inline literals
across voice_pipeline.py or voice.py -- change values here only.

Wake word engine: openWakeWord (local, no API key required).
Model ships pre-downloaded via openwakeword.utils.download_models().
"""

# ── Wake word (openWakeWord) ───────────────────────────────────────────────────

# Pre-trained ONNX model name.  "hey_jarvis_v0.1" ships with download_models().
# Change to any other downloaded model slug (e.g. "alexa_v0.1", "hey_mycroft_v0.1").
OWW_MODEL_NAME = "hey_jarvis_v0.1"

# Detection threshold: score must exceed this to count as a wake word trigger.
# 0.5 is the recommended default -- raise to reduce false positives,
# lower to increase sensitivity (at the cost of more false triggers).
OWW_THRESHOLD = 0.5

# Audio chunk size fed to openWakeWord per predict() call.
# 1280 samples @ 16 kHz = 80 ms per chunk.  Do not change unless you change
# the sample rate -- the model was trained on 16 kHz audio.
OWW_CHUNK_SIZE = 1280

# ── Audio ──────────────────────────────────────────────────────────────────────

AUDIO_SAMPLE_RATE = 16000   # Hz -- required by openWakeWord (fixed, do not change)
AUDIO_CHANNELS    = 1       # mono

# ── Voice activation timeout ───────────────────────────────────────────────────

# Seconds to wait for a spoken command after the wake word fires before
# returning to wake-word listening mode.
VOICE_ACTIVATION_TIMEOUT_SECONDS = 12

# ── Speech-to-text (STT) ───────────────────────────────────────────────────────

# Max seconds to wait for speech to begin before timing out.
STT_LISTEN_TIMEOUT_SECONDS = 4

# Max seconds of a single phrase to capture.
STT_PHRASE_TIME_LIMIT_SECONDS = 8
