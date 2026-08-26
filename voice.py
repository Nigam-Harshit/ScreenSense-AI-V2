"""
voice.py  --  Speech-to-text capture using Google Web Speech (free endpoint).

listen() is called by voice_pipeline.py after wake-word detection.
Returns the recognised command string (lowercased), or "" on any failure.
All failures are logged distinctly to console -- the bare except is gone.
All tuning constants (timeouts) are in voice_config.py.
"""

import speech_recognition as sr

from voice_config import STT_LISTEN_TIMEOUT_SECONDS, STT_PHRASE_TIME_LIMIT_SECONDS


def listen() -> str:

    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = r.listen(
                source,
                timeout=STT_LISTEN_TIMEOUT_SECONDS,
                phrase_time_limit=STT_PHRASE_TIME_LIMIT_SECONDS,
            )
        except sr.WaitTimeoutError:
            # No speech detected within the timeout — expected, not an error.
            print("[STT] No speech detected (timeout).")
            return ""

    try:
        command = r.recognize_google(audio)
        print("Voice command:", command)
        return command.lower()

    except sr.UnknownValueError:
        # Audio captured but unintelligible — common, not a system error.
        print("[STT] Could not understand audio.")
        return ""

    except sr.RequestError as e:
        # Network or Google Speech API failure — worth surfacing clearly.
        print(f"[STT] Network/service error: {e}")
        return ""

    except Exception as e:
        # Catch-all for unexpected failures — log type and message.
        print(f"[STT] Unexpected error ({type(e).__name__}): {e}")
        return ""