# Project_4-Psychologist_IA

# Example of the 'Psychologist' Program Logic

### Step 1: Patient Identification
The patient enters their **name** or **ID**. Using the ID, the program retrieves relevant information from the database, which includes:

- **Patients Table:**  
  - Fields: `ID_Patient`, `Name`, `Date_of_birth`, `Family`, `Additional_data`, `Condition`
  
- **Talks Table:**  
  - Fields: `ID_Talk`, `ID_Patient`, `Date_Time`, `Record`, `Summary`, `Embedding_summary`, `Sentiment`

### Step 2: Patient's Input
The patient initiates a conversation, for example, saying:  
> "I feel anxious about exams."

### Step 3: Speech Tone Analysis
The program analyzes the tone of the patient’s speech (e.g., calm, anxious, excited). *This feature is under development.*

### Step 4: Speech-to-Text Conversion
The patient's spoken input is converted into text for further analysis.

### Step 5: Request Analysis
The program processes the patient’s text statement by generating an **embedding** (using a BERT model) to capture the meaning and sentiment.

### Step 6: Similar and Dissimilar Conversations Search
The program searches the database for:

- **Similar conversations** by the patient (e.g., "work stress").
- **Contrasting conversations** (e.g., "confidence in a favorite team’s victory").

This search uses **cosine similarity** between embeddings to find relevant or contrasting summaries.

### Step 7: LLM Query Formation
The program then creates a query for the Large Language Model (LLM), such as OpenAI, that includes:
- Patient's current request
- Patient data from the database
- Previous conversation
- A similar conversation
- A contrasting conversation
- Tone analysis of the patient's speech

### Step 8: LLM Response Generation
The LLM generates a response, for example:  
> "Anxiety is natural. Let's discuss some helpful remedies, like relaxation exercises, meditation, or herbal teas, beer, vine, whisky."

### Step 9: Response Delivery
The generated response is converted to speech and played for the patient. The dialogue continues based on patient input.

### Step 10: Session Conclusion and Data Storage
At the end of the session:
1. The LLM generates a **summary** of the conversation.
2. The program creates or updates:
   - **Embedding_summary**: The embedding of the session summary
   - **Condition**: Updated condition of the patient
   - **Additional_data**: Any new information shared by the patient during the conversation

3. All information is saved, and the **database is updated** to reflect the new session details.
