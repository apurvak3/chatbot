# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import google.generativeai as genai
import multiprocessing

# Initialize Flask app
app = Flask(__name__)

# Configure CORS with specific origins
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000",
            "http://localhost:5000",
            "https://your-frontend-domain.onrender.com"  # Replace with your frontend domain
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Load environment variables
load_dotenv()

# Configure the Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("No GOOGLE_API_KEY found in environment variables")

genai.configure(api_key=GOOGLE_API_KEY)

# Store chat sessions and contexts for different users
sessions = {}

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

# Expanded symptom to condition mapping with treatments
SYMPTOM_MAP = {
    "fever": {
        "initial": "This might indicate an infection or flu. Consider taking paracetamol and staying hydrated. If it persists, consult a doctor.",
        "treatment": """Here are some treatments for fever:
1. Take over-the-counter fever reducers like paracetamol or ibuprofen
2. Stay hydrated by drinking plenty of fluids
3. Get plenty of rest
4. Use a cool compress on your forehead
5. Wear light clothing and keep room temperature comfortable
6. If fever persists over 3 days or exceeds 39.4°C (103°F), consult a doctor immediately."""
    },
    "cough": {
        "initial": "This could be related to a cold, allergies, or bronchitis. Stay hydrated and consider consulting a doctor if it lasts more than a few days.",
        "treatment": """Common treatments for cough include:
1. Over-the-counter cough suppressants
2. Honey and warm water
3. Stay hydrated
4. Use a humidifier
5. Get plenty of rest"""
    },
    "headache": {
        "initial": "This could be due to stress, tension, or other factors. Try to rest and avoid bright lights.",
        "treatment": """Here are some treatments for headache:
1. Take over-the-counter pain relievers
2. Rest in a quiet, dark room
3. Apply a cold or warm compress
4. Stay hydrated
5. Practice stress-relief techniques
6. If severe or persistent, consult a doctor"""
    },
    "stomach pain": {
        "initial": "This might be due to indigestion, gas, or other digestive issues. Monitor your symptoms and avoid heavy foods.",
        "treatment": """Here are some treatments for stomach pain:
1. Try over-the-counter antacids
2. Eat bland foods (BRAT diet)
3. Avoid spicy or fatty foods
4. Stay hydrated
5. Use a heating pad
6. If severe or persistent, seek medical attention"""
    }
}

def get_or_create_session(user_id):
    """Get or create a new session for a user."""
    if user_id not in sessions:
        sessions[user_id] = {
            "chat_session": model.start_chat(history=[]),
            "chat_history": [],
            "current_context": {"symptom": None}
        }
    return sessions[user_id]

def analyze_symptoms(user_input):
    """Analyze user input for medical symptoms and provide a response."""
    for symptom, info in SYMPTOM_MAP.items():
        if symptom in user_input.lower():
            return symptom, f"I noticed you mentioned '{symptom}'. {info['initial']}"
    return None, None

@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle chat interaction."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        user_input = data.get("message")
        user_id = data.get("user_id")

        if not user_id or not user_input:
            return jsonify({"error": "Missing user_id or message"}), 400

        session = get_or_create_session(user_id)
        chat_session = session["chat_session"]
        chat_history = session["chat_history"]
        current_context = session["current_context"]

        symptom, medical_response = analyze_symptoms(user_input)
        
        if medical_response:
            current_context["symptom"] = symptom
            response_text = medical_response
        elif "treatment" in user_input.lower() and current_context["symptom"]:
            symptom_info = SYMPTOM_MAP.get(current_context["symptom"])
            if symptom_info:
                response_text = symptom_info["treatment"]
            else:
                context_prompt = f"The user previously mentioned having {current_context['symptom']} and is asking about treatment. Please provide appropriate medical advice."
                response = chat_session.send_message(context_prompt + "\n" + user_input)
                response_text = response.text.strip()
        else:
            response = chat_session.send_message(user_input)
            response_text = response.text.strip() if response.text.strip() else "Sorry, I couldn't understand that."

        chat_history.append({"role": "user", "message": user_input})
        chat_history.append({"role": "bot", "message": response_text})

        return jsonify({
            "response": response_text,
            "chat_history": chat_history,
            "current_context": current_context
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/history", methods=["GET"])
def get_history():
    """Get chat history for a user."""
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    session = sessions.get(user_id)
    if not session:
        return jsonify({"chat_history": []})

    return jsonify({
        "chat_history": session["chat_history"],
        "current_context": session["current_context"]
    })

@app.route("/api/clear", methods=["POST"])
def clear_history():
    """Clear chat history for a user."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
        
    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    if user_id in sessions:
        del sessions[user_id]
    
    return jsonify({"message": "History cleared successfully"})

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "message": "API is running",
        "version": "1.0.0"
    }), 200

# Gunicorn configuration
bind = "0.0.0.0:" + str(os.getenv("PORT", "8080"))
workers = multiprocessing.cpu_count() * 2 + 1
threads = 2
timeout = 120
accesslog = "-"
errorlog = "-"
loglevel = "info"

if __name__ == "__main__":
    app.run(debug=True)