# app.py
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

@app.route("/")
def home():
    """Render the home page with chat history."""
    return render_template('chat.html', chat_history=chat_history)

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat interaction."""
    global chat_history
    user_input = request.form.get("user_input")
    
    if user_input:
        try:
            # Start a chat session
            chat_session = model.start_chat(history=[])
            
            # Get response from the model
            response = chat_session.send_message(user_input)
            response_text = response.text.strip()
            
            # Add to chat history
            chat_history.append(("You", user_input))
            chat_history.append(("Bot", response_text))
            
        except Exception as e:
            # Handle any errors
            error_message = f"An error occurred: {str(e)}"
            chat_history.append(("System", error_message))
    
    return render_template('chat.html', chat_history=chat_history)

@app.route("/clear", methods=["POST"])
def clear_history():
    """Clear the chat history."""
    global chat_history
    chat_history = []
    return render_template('chat.html', chat_history=chat_history)

if __name__ == "__main__":
    app.run(debug=True)
