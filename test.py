
def analyze_emotion(audio_data, sr):
    # Convert audio data to float32 and normalize
    audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data), axis=0)

    # Extracting features
    energy = np.mean(librosa.feature.rms(y=audio_data))
    pitch, _ = librosa.piptrack(y=audio_data, sr=sr)
    pitch_mean = np.mean(pitch[pitch > 0])

    # Basic emotion classification
    if energy > 0.1 and pitch_mean > 150:
        emotion = "excited"
    elif energy < 0.05:
        emotion = "calm"
    elif pitch_mean < 120:
        emotion = "scared"
    else:
        emotion = "neutral"
    return emotion


#Speech recognition function with emotion analysis

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
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source)
                full_text = []
                emotion = "No emotion detected"
                while recording and not finish_session:
                    try:
                        audio = r.listen(source, timeout=None)
                        text = r.recognize_google(audio, language=language_full)  # "ru-RU", "en-US" or ""he-IL"
                        print(f"You said: {text}")
                        patient_query += ' ' + text
                        # Convert to audio array for emotion analysis
                        if text:
                             audio_data = np.frombuffer(audio.get_raw_data(), np.int16)
                             emotion = analyze_emotion(audio_data, source.SAMPLE_RATE)
                             print(f"Emotion: {emotion}")

                    except sr.UnknownValueError:
                        print("Could not understand the audio. Please speak clearly.")
                    except sr.RequestError as e:
                        print(f"Service error; {e}")
                        break

                print("Recording stopped.")
                recording = False
                return patient_query, emotion  # Return recognized text with emotion

    return patient_query, "No emotion detected"  # Default return in case of session end