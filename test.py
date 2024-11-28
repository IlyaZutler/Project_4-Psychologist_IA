def recognize_speech(language_full):
    global recording, finish_session
    patient_query = ""
    r = sr.Recognizer()
    print("Press 's' to start recording, 'f' to stop recording. Press 'q' to end the session.")

    while not finish_session:
        if keyboard.is_pressed('s') and not recording:  # Start recording on 's' key press
            recording = True
            print("Recording. Speak...")
            threading.Thread(target=stop_recording, daemon=True).start()  # Thread to stop recording

        if recording:  # Recording is active
            r.adjust_for_ambient_noise(source)
            with sr.Microphone() as source:

                full_text = []
                emotion = "No emotion detected"
                while recording and not finish_session:
                    try:
                        audio = r.listen(source, timeout=None)
                        text = r.recognize_google(audio, language=language_full)  # "ru-RU", "en-US" or ""he-IL"
                        print(f"You said: {text}")
                        patient_query += ' ' + text

                    except sr.UnknownValueError:
                        print("Could not understand the audio. Please speak clearly.")
                    except sr.RequestError as e:
                        print(f"Service error; {e}")
                        break

                print("Recording stopped.")
                recording = False
                return patient_query, emotion  # Return recognized text with emotion

    return patient_query, "No emotion detected"  # Default return in case of session end
