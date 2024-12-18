import speech_recognition as sr

from gtts import gTTS #text to speach
from pydub import AudioSegment
from pydub.playback import play

import pyodbc #for DataBase
from sentence_transformers import SentenceTransformer, util #for embeddigg

from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

import keyboard  #start-finish-end
import threading  #simultaneous operation of several task threads

import librosa # for emotions

import torch
import numpy as np

import os #Key OpenAI

import functools # for decorator how long do functions work
import time
from time import sleep

import queue


#from transformers import pipeline

def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  
        result = func(*args, **kwargs)  
        end_time = time.time()  
        execution_time = end_time - start_time  
        print(f"Execution time for '{func.__name__}': {execution_time:.4f}  seconds.")
        return result
    return wrapper

@log_execution_time
def import_llm_models():
    # API key for OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # Define LLM using OpenAI
    llm = ChatOpenAI(api_key=OPENAI_API_KEY,
                     model="ft:gpt-4o-2024-08-06:personal::AXwxYjWD",
                     temperature=0.5) #gpt-4o-2024-08-06 with fine tuning

    # Model for generating text embeddings (very little Sentence-BERT model)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return llm, model

def stop_recording():
    global recording
    keyboard.wait('f')  # Wait for the 'f' key press
    time.sleep(0.1)  # Small delay before stopping recording
    recording = False  # Set the recording flag to false

def finish_session_function():
    global finish_session
    while not finish_session:
        if keyboard.is_pressed('q'):  # Check for 'q' key press
            finish_session = True  # Set the session finish flag
        time.sleep(0.1)  # Short pause to reduce CPU load

# Function to analyze emotion based on audio features
@log_execution_time
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


# Function to generate an audio response
@log_execution_time
def text_to_speech(text, language):    
    tts = gTTS(text=text, lang=language)  # Changed 'ru' to 'en' or 'iw' NOT 'he' !!!!!!!!!!!!!
    tts.save("response.mp3")
    audio = AudioSegment.from_mp3("response.mp3")
    play(audio)

# Function to find the most recent completed conversation (previous_talk)
@log_execution_time
def find_previous_talk(patient_id, cursor):

    cursor.execute("""
        SELECT TOP 1 Record FROM Talks
        WHERE ID_Patient = ? 
        ORDER BY Date_Time DESC
    """, (patient_id,))
    previous_talk = cursor.fetchone()

    return previous_talk[0] if previous_talk else ""

# Function to find the most similar and most dissimilar conversations
@log_execution_time
def find_similar_talks(model, patient_info, query, cursor):
     
    # Fetch all embeddings and summaries for the given patient from the database
    cursor.execute("SELECT Embedding, Summary FROM Talks WHERE ID_Patient = ?", (patient_info["ID_Patient"],))
    past_talks = cursor.fetchall()
    
    # Generate the embedding for the current query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Variables to store the most similar and most dissimilar talks
    most_similar_talk = None
    most_dissimilar_talk = None
    
    highest_similarity = float('-inf')  # Initialize with a very low value
    lowest_similarity = float('inf')    # Initialize with a very high value

    # Loop through each saved embedding from the database
    for talk in past_talks:
        talk_embedding_str = talk[0]  # Extract the embedding as a string
        talk_summary = talk[1]        # Extract the corresponding summary
        
        # Convert the embedding from a string back to a tensor
        talk_embedding = torch.tensor(list(map(float, talk_embedding_str.split(','))))
        
        # Compute cosine similarity between the current query and the saved embeddings
        similarity = util.pytorch_cos_sim(query_embedding, talk_embedding).item()
        
        # Check if this is the most similar talk so far
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_talk = talk_summary
        
        # Check if this is the most dissimilar talk so far
        if similarity < lowest_similarity:
            lowest_similarity = similarity
            most_dissimilar_talk = talk_summary
    
    # Return the summaries of the most similar and most dissimilar talks
    return most_similar_talk, most_dissimilar_talk


# Function to retrieve patient information from the database
@log_execution_time
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

# Function to generate a response using LLM and langchain-groq. For OpenIA model made fine-tuning as Psychologist
@log_execution_time 
def generate_response_llm(llm, session_record, previous_talk, similar_talk, dissimilar_talk, patient_info, emotions):
    system_message = SystemMessage(content=f"""       
        You are a qualified and concise psychologist, helping the Patient discuss their problems. 
        The conversation consists of several questions and answers. 
        You will receive for answer a record of the current conversation - the Patient's last question at the end. 
        You can use the Socratic questioning technique.  
        Avoid long responses. Ask clarifying questions. 
        Analyze the entire conversation from the very beginning, and not just the Patient's last phrase. 
        Pay attention to all the information about the Patient and previous conversations - You can mention this in your questions.
        Do not repeat the interlocutor's question before answering.
    """)
    ai_message = AIMessage(content=f"""
        Speak {patient_info['Language']} 
        Patient information:
            Name: {patient_info['Name']},
            Date of birth: {patient_info['Date_of_birth']},
            Sex: {patient_info['Sex']},
            Patient_Facts: {patient_info['Patient_Facts']},
            Language: {patient_info['Language']},
            Diagnosis: {patient_info['Diagnosis']},

        Previous conversation with the Patient: {previous_talk or 'No previous conversation'}. 
        Summary of the most similar conversation with the Patient: {similar_talk or 'No similar conversation'}.
        Summary of the most dissimilar conversation with the Patient: {dissimilar_talk or 'No dissimilar conversation'}. 
        Pay attention to the emotional analysis of speeches: {emotions or 'No emotion detected'}      
    """)        
    human_message = HumanMessage(content=f"""
        Here is the current conversation record with the Patient: {session_record},
        the Patient's last question at the end.
    """)
    response = None
    try:
        response = llm.invoke([system_message, ai_message, human_message])
    except Exception as e:
        print(f"Error calling LLM: {e}")      
    
    return response.content


# Function to generate a summary at the end of the conversation
@log_execution_time
def generate_summary(llm, session_record, patient_info):
    system_message = SystemMessage(content=f"""
        You are a qualified Psychologist. 
        Create a brief summary of your conversation with Patient.          
        Don't write introductory words.
        """)
    human_message = HumanMessage(content=f"Conversation: {session_record}.")

    response = llm.invoke([system_message, human_message])
    return response.content

# Function extracts facts about the patient from his conversation and updates the collected data about him
@log_execution_time
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

    # Call the model to update additional_data
    updated_additional_datas = llm.invoke([system_message, human_message])

    # Update additional_datas field using model response
    patient_info['Patient_Facts'] = updated_additional_datas.content.strip()
    return patient_info

@log_execution_time
def update_diagnosis(llm, session_record, patient_info):
    system_message = SystemMessage(content=f"""        
        You are Psychologist and will receive the your conversation transcript with Patient.
        Based on the transcript, infer the patient's possible mental or emotional condition (if identifiable) and provide a concise diagnostic hypothesis for field Diagnosis'.
        If you can't update the Previously known 'Diagnosis' of the Patient, then don't change it, Just return it.
        Don't write introductory words.                
    """)
    ai_message = AIMessage(content=f"""
        Previously known 'Diagnosis' of the Patient: {patient_info['Diagnosis']}.
    """)
    human_message = HumanMessage(
        content=f"Here is the current conversation transcript with the Patient: {session_record}")

    # Call the model to update additional_data
    updated_additional_datas = llm.invoke([system_message, ai_message, human_message])

    # Update additional_datas field using model response
    patient_info['Diagnosis'] = updated_additional_datas.content.strip()

    return patient_info


# Function to save the conversation data, including the embedding, updates Additional_datas into the database
@log_execution_time
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
    print("Please register")
    name = input("Enter name: ")
    language = input("Select language (en, iw, ru): ")
    date_of_birth = input("Enter date of birth (YYYY-MM-DD): ")
    sex = input("Enter sex: ")
    patient_facts = input("Enter additional information if your want: ")
    diagnosis = "No diagnosis inferred"

    # Insert new patient data into the Patients table and get the new ID
    cursor.execute("""
        INSERT INTO Patients (Name, Date_of_birth, Sex, Patient_Facts, Language ) 
        OUTPUT INSERTED.ID_Patient  
        VALUES (?, ?, ?, ?, ?)
    """, (name, date_of_birth, sex, patient_facts, language))

    # Fetch the ID of the new patient record
    patient_id = cursor.fetchone()[0]  # Retrieves the first column of the first row
    
    patient_info = {"ID_Patient": patient_id,
            "Name": name,
            "Date_of_birth": date_of_birth,
            "Sex": sex,
            "Patient_Facts": patient_facts,
            "Language": language,
            "Diagnosis": diagnosis
        }
   
    print(f"Your registered with ID: {patient_id}")
    return patient_info

# Connecting to SQL Server database/ I couldn't open port 1433 and make dynamic DNS in Bezek
# That's why this program is not a useful application 
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

def record_audio(audio_queue, recognizer, language_full):
    """Record audio and put it into the queue."""
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)  # Adjust microphone for ambient noise
        print("Recording started...")
        while recording and not finish_session:
            try:
                # Capture audio segment
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=5)
                audio_queue.put(audio)  # Place audio in the queue
            except Exception as e:
                print(f"Recording error: {e}")
                break
        print("Recording stopped.")

def recognize_audio(audio_queue, recognizer, language_full, result_list, emotion_list):
    """Process audio: recognize speech and analyze emotion."""
    while not finish_session or not audio_queue.empty():
        try:
            # Retrieve audio from the queue
            audio = audio_queue.get(timeout=1)
            # Perform speech recognition
            text = recognizer.recognize_google(audio, language=language_full)
            print(f"You said: {text}")
            result_list.append(text)

            # Convert audio data for emotion analysis
            audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
            emotion = analyze_emotion(audio_data, 16000)  # Replace with the correct sample rate
            emotion_list.append(emotion)
            print(f"Emotion: {emotion}")
        except queue.Empty:
            continue  # Wait for new audio data if the queue is empty
        except sr.UnknownValueError:
            print("Could not understand the audio. Please speak clearly.")
        except sr.RequestError as e:
            print(f"Service error: {e}")
            break

def recognize_speech(language_full):
    """Main function to handle recording and recognizing speech and emotions."""
    global recording, finish_session
    patient_query = []
    emotion_results = []
    audio_queue = queue.Queue()  # Queue to hold audio data
    recognizer = sr.Recognizer()

    print("Press 's' to start recording, 'f' to stop recording. Press 'q' to end the session.")

    while not finish_session:
        # Start recording when 's' is pressed
        if keyboard.is_pressed('s') and not recording:
            recording = True
            print("Recording. Speak...")
            # Start threads for recording and processing
            threading.Thread(target=record_audio, args=(audio_queue, recognizer, language_full), daemon=True).start()
            threading.Thread(target=recognize_audio,
                             args=(audio_queue, recognizer, language_full, patient_query, emotion_results),
                             daemon=True).start()

        # Stop recording when 'f' is pressed
        if keyboard.is_pressed('f') and recording:
            recording = False
            print("Finishing recording...")
            sleep(1)  # Small delay to allow threads to finish

        # End the session when 'q' is pressed
        if keyboard.is_pressed('q'):
            finish_session = True
            print("Ending session...")
            break

    # Combine all recognized text and return the last detected emotion
    final_text = " ".join(patient_query)
    final_emotion = emotion_results[-1] if emotion_results else "No emotion detected"
    return final_text, final_emotion

def main(patient_id):
    global recording, finish_session, llm, model
    recording = False
    finish_session = False
    # Start a thread to monitor session termination
    threading.Thread(target=finish_session_function, daemon=True).start()
    
    llm, model = import_llm_models()

    session_record = ""
    patient_info = {}
    emotions = []
    
    # Using DatabaseConnection as a context manager
    with DatabaseConnection() as (conn, cursor):
        while not patient_info:
            if patient_id == 'r':
                patient_info = register_patient(cursor)
                patient_id = patient_info['ID_Patient'] 
            else:
                previous_talk = find_previous_talk(patient_id, cursor)      
                patient_info = get_patient_info(patient_id, cursor)
                if not patient_info:
                    patient_id = input("ID not found. Enter patient ID or 'r' for registration: ")                                        

        print(f"Starting session for: {patient_info['Name']}")

        language = patient_info['Language'] or 'ru'
        language_map = {'ru':'ru-RU', 'en':'en-US', 'iw':'he-IL'}
        language_full = language_map[language]
        emotion = "No emotion detected"
        
        while True:

            patient_query, emotion_l = recognize_speech(language_full)            
            emotion = emotion_l if emotion_l != "No emotion detected" else emotion
            emotions.append(emotion)
            session_record += f"Patient said: {patient_query}. "

            if finish_session:
                print("Session ended.")
                patient_info = update_patient_facts(llm, session_record, patient_info)
                patient_info = update_diagnosis(llm, session_record, patient_info)
                summary = generate_summary(llm, session_record, patient_info)
                print(f"Summary: {summary}")
                print(f"Patient_info: {patient_info}")
                emotions_string = ', '.join(emotions)
                save_talk(model, patient_id, session_record, summary, emotions_string, patient_info, cursor)
                break

            similar_talk, dissimilar_talk = find_similar_talks(model, patient_info, session_record, cursor)
            response_text = generate_response_llm(
                llm, session_record, previous_talk, similar_talk, dissimilar_talk, patient_info, emotions
            )
            session_record += f"Psychologist responded: {response_text}. "
            print(f"Program response: {response_text}")
            text_to_speech(response_text, language)

# Run the main function
if __name__ == "__main__":
    patient_id = input("Enter patient ID or 'r' for registration: ")
    main(patient_id)