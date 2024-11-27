import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import pyodbc
from sentence_transformers import SentenceTransformer, util
from langchain.schema import SystemMessage, HumanMessage
import torch
import numpy as np
import os
from langchain.chat_models import ChatOpenAI
import librosa

# Function to analyze emotion based on audio features
def analyze_emotion(audio_data, sr):
    audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))
    energy = np.mean(librosa.feature.rms(y=audio_data))
    pitch, _ = librosa.piptrack(y=audio_data, sr=sr)
    pitch_mean = np.mean(pitch[pitch > 0])

    if energy > 0.1 and pitch_mean > 150:
        emotion = "excited"
    elif energy < 0.05:
        emotion = "calm"
    elif pitch_mean < 120:
        emotion = "scared"
    else:
        emotion = "neutral"

    return emotion

# Speech recognition function with emotion analysis
def recognize_speech(language_full):
    r = sr.Recognizer()
    patient_query = ""
    emotion = "No emotion detected"

    with sr.Microphone() as source:
        st.info("Recording... Speak now!")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source, timeout=None)
        try:
            text = r.recognize_google(audio, language=language_full)
            st.write(f"You said: {text}")
            patient_query = text

            audio_data = np.frombuffer(audio.get_raw_data(), np.int16)
            emotion = analyze_emotion(audio_data, source.SAMPLE_RATE)
            st.write(f"Emotion: {emotion}")

        except sr.UnknownValueError:
            st.error("Could not understand the audio. Please speak clearly.")
        except sr.RequestError as e:
            st.error(f"Service error; {e}")

    return patient_query, emotion

# Function to generate an audio response
def text_to_speech(text, language):
    tts = gTTS(text=text, lang=language)
    tts.save("response.mp3")
    audio = AudioSegment.from_mp3("response.mp3")
    play(audio)

# Function to find the most recent completed conversation (previous_talk)
def find_previous_talk(patient_id, cursor):
    cursor.execute("""
        SELECT TOP 1 Record FROM Talks
        WHERE ID_Patient = ? 
        ORDER BY Date_Time DESC
    """, (patient_id,))
    previous_talk = cursor.fetchone()
    return previous_talk[0] if previous_talk else ""

# Function to update the session record with patient query
def update_session_record_query(patient_query, session_record):
    session_record += f"Patient said: {patient_query}. "
    return session_record

# Function to update the session record with program response
def update_session_record_response(program_response, session_record):
    session_record += f"Psychologist responded: {program_response}. "
    return session_record

# Function to find the most similar and most dissimilar conversations
def find_similar_talks(llm, model, patient_id, query, cursor):
    cursor.execute("SELECT Embedding, Summary FROM Talks WHERE ID_Patient = ?", (patient_id,))
    past_talks = cursor.fetchall()

    query_s = generate_summary(llm, query)
    query_embedding = model.encode(query_s, convert_to_tensor=True)

    most_similar_talk = None
    most_dissimilar_talk = None
    highest_similarity = float('-inf')
    lowest_similarity = float('inf')

    for talk in past_talks:
        talk_embedding_str = talk[0]
        talk_summary = talk[1]
        talk_embedding = torch.tensor(list(map(float, talk_embedding_str.split(','))))
        similarity = util.pytorch_cos_sim(query_embedding, talk_embedding).item()

        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_talk = talk_summary

        if similarity < lowest_similarity:
            lowest_similarity = similarity
            most_dissimilar_talk = talk_summary

    return most_similar_talk, most_dissimilar_talk

# Function to retrieve patient information from the database
def get_patient_info(patient_id, cursor):
    cursor.execute("""
        SELECT ID_Patient, Name, Date_of_birth, Sex, Additional_datas, Language 
        FROM Patients 
        WHERE ID_Patient = ?
    """, (patient_id,))
    patient_info = cursor.fetchone()

    if patient_info:
        return {
            "ID_Patient": patient_info[0],
            "Name": patient_info[1],
            "Date_of_birth": patient_info[2],
            "Sex": patient_info[3],
            "Additional_datas": patient_info[4],
            "Language": patient_info[5]
        }
    else:
        return None

# Function to generate a response using LLM
def generate_response_llm(llm, session_record, previous_talk, similar_talk, dissimilar_talk, patient_info, emotion):
    system_message = SystemMessage(content=f"""        
        Speak {patient_info['Language']}
        
        Patient information:
            Name: {patient_info['Name']},
            Date of birth: {patient_info['Date_of_birth']},
            Sex: {patient_info['Sex']},
            Additional data: {patient_info['Additional_datas']},
            Language: {patient_info['Language']}.

        Previous conversation with the Patient: {previous_talk or 'No previous conversation'}. 
        Summary of the most similar conversation with the Patient: {similar_talk or 'No similar conversation'}.
        Summary of the most dissimilar conversation with the Patient: {dissimilar_talk or 'No dissimilar conversation'}. 
        Pay attention to the emotional analysis of speech: {emotion or 'No emotion detected'}      
    """)        
    human_message = HumanMessage(content=f"""
        Here is the current conversation record with the Patient: {session_record}.
    """) 

    try:
        response = llm.invoke([system_message, human_message])
    except Exception as e:
        st.error(f"Error calling LLM: {e}")      
    
    return response.content

# Function to generate a summary at the end of the conversation
def generate_summary(llm, session_record):
    system_message = SystemMessage(content="You are a qualified psychologist. Create a brief summary of your conversation with the Patient.")
    human_message = HumanMessage(content=f"Conversation: {session_record}.")
    
    response = llm([system_message, human_message])
    return response.content

# Function extracts facts about the patient from his conversation and updates the collected data about him
def update_patient_info(llm, session_record, patient_info): 
    system_message = SystemMessage(content=f"""        
        Review the current conversation transcript and update the 'Additional_datas' field with new facts about the patient, if any: {patient_info['Additional_datas']}.        
    """) 
    human_message = HumanMessage(content=f"Here is the current conversation record with the Patient: {session_record}")
    
    updated_additional_datas = llm([system_message, human_message])
    patient_info['Additional_datas'] = updated_additional_datas.content.strip()
    
    return patient_info

# Function to save the conversation data, including the embedding, updates Additional_datas into the database
def save_talk(model, patient_id, session_record, summary, emotion, patient_info, cursor):
    embedding = model.encode(summary, convert_to_tensor=True)
    embedding_str = ','.join(map(str, embedding.tolist()))
    
    cursor.execute("""
        INSERT INTO Talks (ID_Patient, Date_Time, Record, Summary, Emotion, Embedding)
        VALUES (?, GETDATE(), ?, ?, ?, ?)
    """, (patient_id, session_record, summary, emotion, embedding_str))

    Additional_datas = patient_info['Additional_datas']  

    cursor.execute("""
        UPDATE Patients
        SET Additional_datas = ?
        WHERE ID_Patient = ?
    """, (Additional_datas, patient_id))    

# Function to register a new patient
def register_patient(cursor):
    st.write("Please register")
    name = st.text_input("Enter name:")
    language = st.selectbox("Select language:", ['en', 'he', 'ru'])
    date_of_birth = st.text_input("Enter date of birth (YYYY-MM-DD):")
    sex = st.text_input("Enter sex:")
    additional_data = st.text_area("Enter additional information if you want:")    

    if st.button("Register"):
        cursor.execute("""
            INSERT INTO Patients (Name, Date_of_birth, Sex, Additional_datas, Language) 
            OUTPUT INSERTED.ID_Patient
            VALUES (?, ?, ?, ?, ?)
        """, (name, date_of_birth, sex, additional_data, language))    

        patient_id = cursor.fetchone()[0]
        st.write(f"Registered with ID: {patient_id}")
        return {
            "ID_Patient": patient_id,
            "Name": name,
            "Date_of_birth": date_of_birth,
            "Sex": sex,
            "Additional_datas": additional_data,
            "Language": language
        }
    return None

# Connecting to SQL Server database
class DatabaseConnection:
    def __enter__(self):
        self.conn = pyodbc.connect(
            'DRIVER={SQL Server};'
            'SERVER=DESKTOP-SPMU70G\\SQLEXPRESS;'
            'DATABASE=Psychologist;'
            'UID=DESKTOP-SPMU70G\\domashniy;'
            'Trusted_Connection=yes;'
        )
        self.cursor = self.conn.cursor()
        return self.conn, self.cursor

    def __exit__(self, exc_type, exc_value, traceback):
        self.conn.commit()
        self.conn.close()

def import_llm_models():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="ft:gpt-4o-2024-08-06:personal:psychologist-1:APyJnbej", temperature=0.5)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return llm, model

def main():
    st.title("Psychologist Session")
    patient_id = st.text_input("Enter patient ID or 'r' for registration:")

    if not patient_id:
        st.warning("Please enter a patient ID or 'r' to register.")
        return

    llm, model = import_llm_models()
    session_record = ""
    response_text = ""
    recording = False

    with DatabaseConnection() as (conn, cursor):
        if patient_id.lower() == 'r':
            patient_info = register_patient(cursor)
            if not patient_info:
                st.warning("Please complete the registration form.")
                return
            patient_id = patient_info['ID_Patient']
        else:
            previous_talk = find_previous_talk(patient_id, cursor)
            patient_info = get_patient_info(patient_id, cursor)
            if not patient_info:
                st.error("Patient not found. Please register.")
                return

        st.write(f"Starting session for: {patient_info['Name']}")

        similar_talk = dissimilar_talk = ""
        language = patient_info['Language'] or 'ru'
        language_map = {'ru': 'ru-RU', 'en': 'en-US', 'he': 'he-IL'}
        language_full = language_map[language]

        emotion = "No emotion detected"

        if st.button("Start Recording"):
            recording = True

        if st.button("Stop Recording"):
            recording = False

        if recording:
            patient_query, emotion_l = recognize_speech(language_full)
            emotion = emotion_l if emotion_l != "No emotion detected" else emotion

            session_record = update_session_record_query(patient_query, session_record)

            if not similar_talk and not dissimilar_talk:
                similar_talk, dissimilar_talk = find_similar_talks(
                    llm, model, patient_id, session_record, cursor
                )

            response_text = generate_response_llm(
                llm, session_record, previous_talk, similar_talk, dissimilar_talk, patient_info, emotion
            )
            st.write(f"Program response: {response_text}")

            session_record = update_session_record_response(response_text, session_record)

            text_to_speech(response_text, language)

        if st.button("End Session"):
            summary = generate_summary(llm, session_record)
            st.write(f"Conversation summary: {summary}")
            update_patient_info(llm, session_record, patient_info)
            save_talk(model, patient_id, session_record, summary, emotion, patient_info, cursor)
            st.success("Session ended and data saved.")

# Run the main function
if __name__ == "__main__":
    main()