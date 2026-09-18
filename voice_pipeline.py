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

import os
import sys
from pathlib import Path

from voice import listen
from voice_config import (
    OWW_MODEL_NAME,
    OWW_THRESHOLD,
    OWW_CHUNK_SIZE,
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    VOICE_ACTIVATION_TIMEOUT_SECONDS,
)


def _load_oww_model(model_name: str):
    """
    Deterministically resolve and load the openWakeWord model using the package's
    supported resource mechanism.
    If the model is missing in the active environment, attempts automatic acquisition
    via openwakeword.utils.download_models().
    Produces concise, actionable errors if loading cannot proceed.
    """
    try:
        import openwakeword
        from openwakeword.model import Model
    except ImportError as e:
        raise RuntimeError(
            f"\n[Voice Error] openWakeWord is not installed in the active environment:\n"
            f"  Python: {sys.executable}\n"
            f"  Run: python -m pip install openwakeword\n"
        ) from e

    target_onnx = f"{model_name}.onnx" if not model_name.endswith(".onnx") else model_name
    model_slug = model_name[:-5] if model_name.endswith(".onnx") else model_name

    # 1. Resolve path using openwakeword's official pretrained models directory
    pretrained_paths = openwakeword.get_pretrained_model_paths("onnx")
    matched_paths = [p for p in pretrained_paths if os.path.basename(p) == target_onnx]

    resolved_path = None
    if matched_paths and os.path.exists(matched_paths[0]):
        resolved_path = matched_paths[0]
    else:
        # Check project-relative models directory as fallback
        local_model = Path("models") / target_onnx
        if local_model.exists():
            resolved_path = str(local_model.resolve())

    # 2. If model file is missing, attempt one-time supported acquisition
    if not resolved_path:
        expected_path = matched_paths[0] if matched_paths else os.path.join(
            os.path.dirname(openwakeword.__file__), "resources", "models", target_onnx
        )
        print(f"[Voice] Wake word model '{model_name}' not found at: {expected_path}")
        print(f"[Voice] Attempting automatic acquisition via openwakeword.utils.download_models()...")
        try:
            from openwakeword.utils import download_models
            download_models(model_names=[model_slug])
            if os.path.exists(expected_path):
                print(f"[Voice] Successfully acquired '{model_name}' -> {expected_path}")
                resolved_path = expected_path
            else:
                print(f"[Voice] Model file still missing after download attempt.")
        except Exception as dl_err:
            print(f"[Voice] Automatic model acquisition failed: {dl_err}")

    # 3. Final verification and loading
    if not resolved_path or not os.path.exists(resolved_path):
        expected_path = matched_paths[0] if matched_paths else os.path.join(
            os.path.dirname(openwakeword.__file__), "resources", "models", target_onnx
        )
        raise RuntimeError(
            f"\n[Voice Error] Unable to load openWakeWord model '{model_name}'.\n"
            f"  - Model Name:     {model_name}\n"
            f"  - Expected Path:  {expected_path}\n"
            f"  - Interpreter:    {sys.executable}\n"
            f"  - Package File:   {openwakeword.__file__}\n"
            f"  - Required Action: Run the following command inside your environment:\n"
            f"      python -c \"from openwakeword.utils import download_models; download_models()\"\n"
        )

    try:
        oww_model = Model(
            wakeword_models=[resolved_path],
            inference_framework="onnx",
        )
        return oww_model
    except Exception as e:
        raise RuntimeError(
            f"[Voice] Failed to load openWakeWord model '{OWW_MODEL_NAME}': {e}\n"
            "        Run: python -c \"from openwakeword.utils import download_models; download_models()\"\n"
            "        Then restart."
            f"\n[Voice Error] Failed initializing ONNX session for '{resolved_path}': {e}\n"
            f"  - Interpreter:  {sys.executable}\n"
            f"  - Package File: {openwakeword.__file__}\n"
        ) from e


def voice_loop(run_command):
    """
    Main voice loop.  Listens continuously for the wake word, then captures
    and dispatches a spoken command.  Loops forever until the process exits.
    """
    oww_model = _load_oww_model(OWW_MODEL_NAME)
    model_key = OWW_MODEL_NAME[:-5] if OWW_MODEL_NAME.endswith(".onnx") else OWW_MODEL_NAME

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
            score = prediction.get(model_key, prediction.get(OWW_MODEL_NAME, 0.0))

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