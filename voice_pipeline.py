"""
voice_pipeline.py  --  Wake-word loop + voice command dispatch.

Uses openWakeWord (local ONNX, no API key) for wake-word detection ("hey jarvis")
and voice.listen() for STT after activation.

Wake word: say "hey jarvis" to activate.
All tuning constants are in voice_config.py.
No API key or cloud account required.
"""

import numpy as np
import sounddevice as sd
import queue
import threading
import time

from voice import listen
from voice_config import (
    OWW_MODEL_NAME,
    OWW_THRESHOLD,
    OWW_CHUNK_SIZE,
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    VOICE_ACTIVATION_TIMEOUT_SECONDS,
)


def voice_loop(run_command):
    """
    Main voice loop.  Listens continuously for the wake word, then captures
    and dispatches a spoken command.  Loops forever until the process exits.
    """
    # Load the wake word model (ONNX backend, no tflite needed)
    try:
        from openwakeword.model import Model
        oww_model = Model(
            wakeword_models=[OWW_MODEL_NAME],
            inference_framework="onnx",
        )
    except Exception as e:
        raise RuntimeError(
            f"[Voice] Failed to load openWakeWord model '{OWW_MODEL_NAME}': {e}\n"
            "        Run: python -c \"from openwakeword.utils import download_models; download_models()\"\n"
            "        Then restart."
        ) from e

    audio_q: queue.Queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        # indata shape: (frames, channels) -- flatten to 1D int16
        audio_q.put(indata[:, 0].copy())

    print("Voice system ready. Say 'hey jarvis' to activate.")

    stream = sd.InputStream(
        samplerate=AUDIO_SAMPLE_RATE,
        blocksize=OWW_CHUNK_SIZE,
        dtype="int16",
        channels=AUDIO_CHANNELS,
        callback=audio_callback,
    )

    with stream:
        while True:
            # ── Wake-word listening phase ──────────────────────────────────────
            chunk = audio_q.get()
            prediction = oww_model.predict(chunk)
            score = prediction.get(OWW_MODEL_NAME, 0.0)

            if score < OWW_THRESHOLD:
                continue

            # ── Wake word detected ─────────────────────────────────────────────
            print(f"Wake word detected! (score={score:.2f})")

            # Drain any stale audio that queued up during inference
            while not audio_q.empty():
                audio_q.get_nowait()

            # Temporarily pause the stream so sounddevice mic is free for STT
            stream.stop()
            try:
                command = listen()
                if command:
                    run_command(command)
                else:
                    print("[Voice] No command heard — returning to wake mode.")
            finally:
                # Always restart listening, even if run_command raises
                oww_model.reset()   # clear internal state between activations
                stream.start()
                print("Listening for wake word...")