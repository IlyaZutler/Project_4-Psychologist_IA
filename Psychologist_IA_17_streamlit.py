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
from langchain_openai import ChatOpenAI
import librosa

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

    query_embedding = model.encode(query, convert_to_tensor=True)

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
        SELECT ID_Patient, Name, Date_of_birth, Sex, Patient_Facts, Language, Diagnosis
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
            "Patient_Facts": patient_info[4],
            "Language": patient_info[5],
            "Diagnosis": patient_info[6]
        }
    else:
        return None

# Function to generate a response using LLM
def generate_response_llm(llm, session_record, previous_talk, similar_talk, dissimilar_talk, patient_info, emotion):
    system_message = SystemMessage(content=f"""        
        Speak {patient_info['Language']}
        You are a qualified and concise psychologist, helping the Patient discuss their problems. 
        The conversation consists of several questions and answers. 
        You will receive for anser a record of the current conversation - the Patient's last question at the end. 
        You can use the Socratic questioning technique.  
        Avoid long responses. Ask clarifying questions. 
        Analyze the entire conversation from the very beginning, and not just the Patient's last phrase. 
        Pay attention to all the information about the Patient and previous conversations - You can mention this in your questions.
        Do not repeat the interlocutor's question before answering.
        Patient information:
            Name: {patient_info['Name']},
            Date of birth: {patient_info['Date_of_birth']},
            Sex: {patient_info['Sex']},
            Patient_Facts: {patient_info['Patient_Facts']},
            Language: {patient_info['Language']}
            Diagnosis: {patient_info['Diagnosis']},
     
    """)        
    human_message = HumanMessage(content=f"""
        Here is the current conversation record with the Patient: {session_record}.
        Previous conversation with the Patient: {previous_talk or 'No previous conversation'}. 
        Summary of the most similar conversation with the Patient: {similar_talk or 'No similar conversation'}.
        Summary of the most dissimilar conversation with the Patient: {dissimilar_talk or 'No dissimilar conversation'}. 
        Pay attention to the emotional analysis of speech: {emotion or 'No emotion detected'}
    """) 

    try:
        response = llm.invoke([system_message, human_message])
    except Exception as e:
        st.error(f"Error calling LLM: {e}")      
    
    return response.content

# Function to generate a summary at the end of the conversation
def generate_summary(llm, session_record, patient_info):
    system_message = SystemMessage(content=f"""
        You are a qualified psychologist. 
        Create a brief summary of your conversation with Patient.
        Don't write introductory words.
        """)
    human_message = HumanMessage(content=f"Here is conversation: {session_record}.")
    
    response = llm.invoke([system_message, human_message])
    return response.content

# Function extracts facts about the patient from his conversation and updates the collected data about him
def update_patient_facts(llm, session_record, patient_info):
    system_message = SystemMessage(content=f"""     
        You are Psychologist. You will receive known Patient facts.    
        You will receive the conversation transcript between the Psychologist and the Patient.            
        Review the transcript and extract new facts about the Patient, if any. 
        Extract only relevant information and avoid redundant details. 
        You need return information for field 'Patient_Facts' which will include previously known Patient facts and new facts. Or update Previously known facts if necessary.         
        This is not a diagnosis and not a description of the patient's condition.
        Don't write introductory words.
        """)

    human_message = HumanMessage(content=f"""
        Here is known Patient facts: {patient_info['Patient_Facts']};
        Here is the current conversation transcript with the Patient: {session_record}
        """)
    
    updated_additional_datas = llm.invoke([system_message, human_message])
    patient_info['Patient_Facts'] = updated_additional_datas.content.strip()
    
    return patient_info

def update_diagnosis(llm, session_record, patient_info):
    system_message = SystemMessage(content=f"""        
        You are Psychologist and will receive the your conversation transcript with Patient.
        Based on the transcript, infer the patient's possible mental or emotional condition (if identifiable) and provide a concise diagnostic hypothesis for field Diagnosis'.
        Previously known 'Diagnosis' of the Patient: {patient_info['Diagnosis']}.
        Don't write introductory words.
    """)

    human_message = HumanMessage(
        content=f"Here is the current conversation transcript with the Patient: {session_record}")

    # Call the model to update additional_data
    updated_additional_datas = llm.invoke([system_message, human_message])

    # Update additional_datas field using model response
    patient_info['Diagnosis'] = updated_additional_datas.content.strip()

    return patient_info

# Function to save the conversation data, including the embedding, updates Additional_datas into the database
def save_talk(model, patient_id, session_record, summary, emotion, patient_info, cursor):
    embedding = model.encode(session_record, convert_to_tensor=True)
    embedding_str = ','.join(map(str, embedding.tolist()))
    
    cursor.execute("""
        INSERT INTO Talks (ID_Patient, Date_Time, Record, Summary, Emotion, Embedding)
        VALUES (?, GETDATE(), ?, ?, ?, ?)
    """, (patient_id, session_record, summary, emotion, embedding_str))

    patient_facts, diagnosis = patient_info['Patient_Facts'], patient_info['Diagnosis']

    cursor.execute("""
        UPDATE Patients
        SET Patient_Facts = ?, Diagnosis = ? 
        WHERE ID_Patient = ?
    """, (patient_facts, diagnosis, patient_id))

# Function to register a new patient
def register_patient(cursor):
    st.write("Please register")
    name = st.text_input("Enter name:")
    language = st.selectbox("Select language:", ['en', 'iw', 'ru'])
    date_of_birth = st.text_input("Enter date of birth (YYYY-MM-DD):")
    sex = st.text_input("Enter sex:")
    patient_facts = st.text_area("Enter additional information if you want:")
    diagnosis = "No diagnosis inferred"

    if st.button("Register"):
        cursor.execute("""
            INSERT INTO Patients (Name, Date_of_birth, Sex, Patient_Facts, Language) 
            OUTPUT INSERTED.ID_Patient
            VALUES (?, ?, ?, ?, ?)
        """, (name, date_of_birth, sex, patient_facts, language))

        patient_id = cursor.fetchone()[0]
        st.write(f"Registered with ID: {patient_id}")
        return {
            "ID_Patient": patient_id,
            "Name": name,
            "Date_of_birth": date_of_birth,
            "Sex": sex,
            "Patient_Facts": patient_facts,
            "Language": language,
            "Diagnosis": diagnosis
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
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, 
                     model="ft:gpt-4o-2024-08-06:personal::AXwxYjWD",
                     temperature=0.5) #gpt-4o-2024-08-06 with fine tuning    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return llm, model

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

def recognize_speech(language_full):
    r = sr.Recognizer()
    st.session_state.patient_query = ""
    st.session_state.full_text = []
    st.session_state.emotion = "No emotion detected"
    # Only start the microphone if the recording flag is True  
    
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)            
        while st.session_state.recording:
                st.info("Recording... Speak now!")
                try:
                    audio = r.listen(source, timeout=None)
                    text = r.recognize_google(audio, language=language_full)
                    st.write(f"You said: {text}")
                    st.session_state.full_text.append(text)
                    if text:                                               
                        audio_data = np.frombuffer(audio.get_raw_data(), np.int16)                        
                        st.session_state.emotion = analyze_emotion(audio_data, source.SAMPLE_RATE) #22050 
                        st.write(f"Emotion: {st.session_state.emotion}")
                except sr.UnknownValueError:
                    continue  # Still listening, do not stop on unknown value
        st.write("Position 1:")
        st.session_state.patient_query = ' '.join(st.session_state.full_text)            
        st.write(f"Full patient query 2: {st.session_state.patient_query}")   
        return st.session_state.patient_query, st.session_state.emotion


def main():

    st.title("Psychologist Session")
    
    patient_id = st.text_input("Enter your ID or 'r' for registration:", key="patient_id_input")

    if not patient_id:
        st.warning("Please enter a your ID or 'r' to register.")
        return

    if "llm" not in st.session_state or "model" not in st.session_state:
        st.write("Loading models...")
        llm, model = import_llm_models()
        st.session_state.llm = llm
        st.session_state.model = model
    else:
        llm = st.session_state.llm
        model = st.session_state.model

    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'was_record' not in st.session_state:
        st.session_state.have_record = False
    if 'end_session' not in st.session_state:
        st.session_state.end_session = False       
    # st.session_state.setdefault("was_record", True)

    st.session_state.setdefault("language", "")
    st.session_state.setdefault("language_map", "")
    st.session_state.setdefault("language_full", "")

    st.session_state.setdefault("session_record", "")
    st.session_state.setdefault("patient_query", "")
    st.session_state.setdefault("response_text", "")
    st.session_state.setdefault("patient_info", "")
    

    with DatabaseConnection() as (conn, cursor):
        if patient_id.lower() == 'r':
            st.session_state.patient_info = register_patient(cursor)
            if not st.session_state.patient_info:
                st.warning("Please complete the registration form.")
                return
            st.session_state.patient_id = st.session_state.patient_info['ID_Patient']
        else:
            previous_talk = find_previous_talk(patient_id, cursor)
            st.session_state.patient_info = get_patient_info(patient_id, cursor)
            if not st.session_state.patient_info:
                st.error("ID not found. Please register.")
                return

        st.write(f"Starting session for: {st.session_state.patient_info['Name']}")

        st.session_state.language = st.session_state.patient_info['Language'] or 'ru'
        st.session_state.language_map = {'ru': 'ru-RU', 'en': 'en-US', 'iw': 'he-IL'}
        st.session_state.language_full = st.session_state.language_map[st.session_state.language]

        st.session_state.emotion = "No emotion detected"
        
        if st.button("Start Recording"):
            st.session_state.recording = True
            st.session_state.was_record = False

        if st.button("Stop Recording"):
            st.session_state.recording = False
            st.session_state.have_record = True
        
        emotion_local = ""
        if st.session_state.recording:
            st.write("Position 4:")
            st.session_state.patient_query, emotion_local = recognize_speech(st.session_state.language_full)
            
        emotion = emotion_local if emotion_local != "No emotion detected" else emotion
        
        st.write(f"patient_query 5: {st.session_state.patient_query}")
        st.write("Position 6")

        if st.session_state.have_record and not st.session_state.recording and not st.session_state.end_session:
            st.write(f"Position 7:")
            st.session_state.session_record = update_session_record_query(st.session_state.patient_query, st.session_state.session_record)            
           
            st.write(f"Session record 8: {st.session_state.session_record}")
            st.write(st.session_state.session_record)
            similar_talk, dissimilar_talk = find_similar_talks(llm, model, patient_id, st.session_state.session_record, cursor)            
            response_text = generate_response_llm(llm, st.session_state.session_record, previous_talk, similar_talk, dissimilar_talk, st.session_state.patient_info, emotion)            
            st.write(f"Program response: {response_text}")
            st.session_state.session_record = update_session_record_response(response_text, st.session_state.session_record)
            text_to_speech(response_text, st.session_state.language)
            
            #st.session_state.was_record = False

        if st.button("End Session"):
            st.session_state.end_session = True

        if st.session_state.end_session:
            st.session_state.patient_info = update_patient_facts(llm, st.session_state.session_record, st.session_state.patient_info)
            st.session_state.patient_info = update_diagnosis(llm, st.session_state.session_record, st.session_state.patient_info)
            summary = generate_summary(llm, st.session_state.session_record, st.session_state.patient_info)
            st.write(f"Conversation summary: {summary}")
            save_talk(model, patient_id, st.session_state.session_record, summary, emotion, st.session_state.patient_info, cursor)
            st.success("Session ended and data saved.")

# Run the main function
if __name__ == "__main__":
    main()


