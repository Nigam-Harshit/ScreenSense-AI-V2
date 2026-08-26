"""
voice_pipeline.py  --  Wake-word loop + voice command dispatch.

Uses pvporcupine for wake-word detection ("jarvis") and voice.listen()
for STT after activation.  All tuning constants are in voice_config.py.

Porcupine access key must be set in .env as PORCUPINE_ACCESS_KEY.
"""

import pvporcupine
import sounddevice as sd
import struct
import time

from voice import listen
from voice_config import (
    WAKE_KEYWORD,
    VOICE_ACTIVATION_TIMEOUT_SECONDS,
    AUDIO_CHANNELS,
    load_porcupine_key,
)


def voice_loop(run_command):

    access_key = load_porcupine_key()
    if not access_key:
        raise RuntimeError(
            "[Voice] PORCUPINE_ACCESS_KEY is not set.\n"
            "        Add it to your .env file:  PORCUPINE_ACCESS_KEY=your_key_here\n"
            "        Or run in text mode (T) instead."
        )

    porcupine = pvporcupine.create(
        access_key=access_key,
        keywords=[WAKE_KEYWORD],
    )

    mode = "wake"
    last_command_time = 0

    print("Voice system ready...")

    def audio_callback(indata, frames, time_info, status):

        nonlocal mode, last_command_time

        pcm = struct.unpack_from("h" * porcupine.frame_length, indata)

        if mode == "wake":

            keyword_index = porcupine.process(pcm)

            if keyword_index >= 0:
                print("Wake word detected!")
                mode = "command"
                last_command_time = time.time()

        elif mode == "command":

            if time.time() - last_command_time > VOICE_ACTIVATION_TIMEOUT_SECONDS:
                print("Returning to wake mode...")
                mode = "wake"
                return

    stream = sd.RawInputStream(
        samplerate=porcupine.sample_rate,
        blocksize=porcupine.frame_length,
        dtype="int16",
        channels=AUDIO_CHANNELS,
        callback=audio_callback,
    )
    stream.start()

    try:
        while True:

            if mode == "command":

                stream.stop()

                command = listen()

                if command:
                    run_command(command)

                print("Returning to wake mode...")
                mode = "wake"
                stream.start()

            time.sleep(0.1)
    finally:
        stream.stop()
        stream.close()