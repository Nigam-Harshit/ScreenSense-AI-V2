import pvporcupine
import sounddevice as sd
import struct
import time
from voice import listen

ACCESS_KEY = "DGVosSJONibo052lrjN4iMeO/PJqvTR4NBtvVVv5nNU/w3uV/OrsVA=="

activation_timeout = 12

def voice_loop(run_command):

    porcupine = pvporcupine.create(
        access_key="DGVosSJONibo052lrjN4iMeO/PJqvTR4NBtvVVv5nNU/w3uV/OrsVA==",
        keywords=["jarvis"]
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

            if time.time() - last_command_time > activation_timeout:
                print("Returning to wake mode...")
                mode = "wake"
                return

    stream = sd.RawInputStream(
        samplerate=porcupine.sample_rate,
        blocksize=porcupine.frame_length,
        dtype="int16",
        channels=1,
        callback=audio_callback
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