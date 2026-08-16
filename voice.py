import speech_recognition as sr

def listen():

    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = r.listen(source, timeout=4, phrase_time_limit=8)
        except sr.WaitTimeoutError:
            return ""

    try:
        command = r.recognize_google(audio)
        print("Voice command:", command)
        return command.lower()

    except Exception:
        return ""