# Project_4-Psychologist_IA

Example of the 'Psychologist' Program Logic:

The patient enters their name or ID.
Using the ID, the program retrieves data from the database, which includes:

Patients Table: Contains ID_Patient, Name, Date_of_birth, Family, Additional_data, Condition.
Talks Table: Contains ID_Talk, ID_Patient, Date_Time, Record, Summary, Embedding_summary, Sentiment.
Patient's Input: The patient says, "I feel anxious about exams."

Tone Analysis: The program analyzes the tone of the speech—calm, anxious, etc. (currently under development).

Speech-to-Text Conversion: The patient's statement is converted to text.

Request Analysis: The program performs content analysis of the patient’s statement by generating an embedding (using a BERT model).

Similar Conversations Search: The program searches the database for similar past talks by the patient (e.g., "work stress") and contrasting dialogues (e.g., "confidence in a favorite team’s victory") based on cosine similarity between embeddings of summaries.

LLM Query Formation: The program then creates a query for the LLM (OpenAI or equivalent), which includes the patient’s current request, patient data, previous conversation, a similar conversation, a contrasting conversation, and tone analysis.

LLM Response Generation: The LLM generates a response, e.g., "Anxiety is natural. Let's discuss some possible remedies: herbal tea, meditation, or relaxation techniques."

Response Delivery: The program converts the response to speech and delivers it to the patient. The dialogue continues.

Session Conclusion: At the end of the conversation, the LLM creates a summary, embedding_summary, updates the patient’s Condition, and adds any new information to Additional_data.

Data Storage: All information is saved, and the database is updated.
