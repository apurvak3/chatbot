from flask import Flask, render_template, request
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Store chat history globally
chat_history = []

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

@app.route("/")
def home():
    """Render the home page with chat history."""
    # Ensure chat_history is passed as a list of dictionaries
    history_for_render = [{"role": item[0], "content": item[1]} for item in chat_history]
    return render_template("chat.html", chat_history=history_for_render)

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat interaction."""
    global chat_history
    user_input = request.form.get("user_input")
    
    if user_input:
        try:
            # Log user input for debugging
            print(f"User input: {user_input}")
            
            # Check for symptoms in the input
            medical_response = analyze_symptoms(user_input)
            
            if medical_response:
                # Use tailored medical response
                response_text = medical_response
            else:
                # Use the Gemini model for general responses
                chat_session = model.start_chat(history=[])
                response = chat_session.send_message(user_input)
                
                # Log the raw response
                print(f"Raw bot response: {response.text}")
                
                if not response.text.strip():  # Check if response is empty or just whitespace
                    response_text = "Sorry, I couldn't understand that."
                else:
                    # Trim or simplify the response
                    response_text = simplify_response(response.text.strip())
            
            # Log the response being added to the history
            print(f"Bot response: {response_text}")
            
            # Append to chat history in a format that the template can easily use
            chat_history.append({"role": "You", "message": user_input})
            chat_history.append({"role": "Bot", "message": response_text})
            
        except Exception as e:
            # Handle any errors
            error_message = f"An error occurred: {str(e)}"
            print(error_message)  # Log the error
            chat_history.append({"role": "System", "message": error_message})
    
    return render_template("chat.html", chat_history=chat_history)
@app.route("/clear", methods=["POST"])
def clear_history():
    """Clear the chat history."""
    global chat_history
    chat_history = []
    return render_template("chat.html", chat_history=chat_history)

def analyze_symptoms(user_input):
    """
    Analyze user input for medical symptoms and provide a response.
    """
    for symptom, advice in SYMPTOM_MAP.items():
        if symptom in user_input.lower():
            return f"I noticed you mentioned '{symptom}'. {advice}."
    return None

def simplify_response(response_text):
    """
    Simplify or summarize the model's response to make it more concise.
    """
    # Example: Take only the first paragraph or key actionable points
    lines = response_text.split("\n")
    return "\n".join(lines[:3]) if len(lines) > 3 else response_text

if __name__ == "__main__":
    app.run(debug=True)
