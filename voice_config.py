"""
voice_config.py  --  Centralised config for voice input (wake word + STT).

All voice-related constants live here.  Do NOT scatter inline literals
across voice_pipeline.py or voice.py -- change values here only.

Porcupine key: read from PORCUPINE_ACCESS_KEY env var or .env file.
The key must NOT be hardcoded in source -- set it in your .env file:

    PORCUPINE_ACCESS_KEY=your_key_here

"""

import os
from pathlib import Path

# ── Wake word ──────────────────────────────────────────────────────────────────

WAKE_KEYWORD = "jarvis"

# Seconds to wait for a spoken command after the wake word fires.
VOICE_ACTIVATION_TIMEOUT_SECONDS = 12

# ── Speech-to-text (STT) ───────────────────────────────────────────────────────

# Max seconds to wait for speech to begin before timing out.
STT_LISTEN_TIMEOUT_SECONDS = 4

# Max seconds of a single phrase to capture.
STT_PHRASE_TIME_LIMIT_SECONDS = 8

# ── Audio ──────────────────────────────────────────────────────────────────────

AUDIO_CHANNELS = 1   # mono

# ── Porcupine access key ───────────────────────────────────────────────────────

PORCUPINE_KEY_ENV = "PORCUPINE_ACCESS_KEY"


def load_porcupine_key() -> str | None:
    """Read Porcupine access key from env var or .env file.
    Returns None if not set -- caller should raise a clear error.
    """
    key = os.environ.get(PORCUPINE_KEY_ENV)
    if key:
        return key
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{PORCUPINE_KEY_ENV}=") and not line.startswith("#"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None
