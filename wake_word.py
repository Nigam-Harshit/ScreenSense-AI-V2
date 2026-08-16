import pvporcupine
import sounddevice as sd
import struct
import time

ACCESS_KEY = "DGVosSJONibo052lrjN4iMeO/PJqvTR4NBtvVVv5nNU/w3uV/OrsVA=="


def wait_for_wake_word():

    porcupine = pvporcupine.create(
        access_key="DGVosSJONibo052lrjN4iMeO/PJqvTR4NBtvVVv5nNU/w3uV/OrsVA==",
        keywords=["jarvis"]
    )

    detected = False

    print("\nSay the wake word...")
    print("Listening for wake word...")

    def audio_callback(indata, frames, time_info, status):
        nonlocal detected

        pcm = struct.unpack_from("h" * porcupine.frame_length, indata)
        keyword_index = porcupine.process(pcm)

        if keyword_index >= 0:
            detected = True

    with sd.RawInputStream(
        samplerate=porcupine.sample_rate,
        blocksize=porcupine.frame_length,
        dtype="int16",
        channels=1,
        callback=audio_callback
    ):

        while not detected:
            time.sleep(0.05)

    print("Wake word detected!\n")

    porcupine.delete()