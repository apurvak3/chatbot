import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure generation settings for the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

# Define the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Symptom to condition mapping
SYMPTOM_MAP = {
    "fever": "This might indicate an infection or flu. Consider taking paracetamol and staying hydrated. If it persists, consult a doctor.",
    "cough": "This could be related to a cold, allergies, or bronchitis. Stay hydrated and consider consulting a doctor if it lasts more than a few days.",
    "headache": "You might be experiencing a migraine or stress-related headache. Rest and hydration may help.",
    "stomach pain": "This might be due to indigestion or gastritis. Avoid heavy meals and monitor your symptoms. Consult a doctor if severe.",
    "sneezing": "Sneezing often indicates an allergy or the onset of a cold. Try to avoid allergens and rest well.",
    "body heating": "A feeling of overheating might be due to fever or dehydration. Ensure you drink enough water and rest."
}

def analyze_symptoms(user_input):
    """
    Analyze user input for medical symptoms and provide a response.
    """
    for symptom, advice in SYMPTOM_MAP.items():
        if symptom in user_input.lower():
            return f"I noticed you mentioned '{symptom}'. {advice}."
    return None

def get_chat_response(user_input):
    """
    Generate a response from the AI model for general queries.
    """
    try:
        # Start a chat session
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(user_input)
        response_text = response.text.strip()
        
        if not response_text:
            return "Sorry, I couldn't understand that."
        return simplify_response(response_text)
    except Exception as e:
        return f"Error: {str(e)}"

def simplify_response(response_text):
    """
    Simplify or summarize the model's response to make it more concise.
    """
    lines = response_text.split("\n")
    return "\n".join(lines[:3]) if len(lines) > 3 else response_text
