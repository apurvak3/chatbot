from flask import Flask, render_template, request, redirect, url_for
from report_analysis import analyze_report
from chatbot_logic import analyze_symptoms, get_chat_response
import os

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Store chat history globally
chat_history = []

@app.route("/")
def home():
    """Render the home page with chat history."""
    return render_template("chat.html", chat_history=chat_history)

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat interaction."""
    global chat_history
    user_input = request.form.get("user_input")
    
    if user_input:
        # Analyze symptoms first
        medical_response = analyze_symptoms(user_input)
        
        if medical_response:
            response_text = medical_response
        else:
            # Get response from the AI model
            response_text = get_chat_response(user_input)
        
        # Add to chat history
        chat_history.append({"role": "You", "message": user_input})
        chat_history.append({"role": "Bot", "message": response_text})
    
    return render_template("chat.html", chat_history=chat_history)

@app.route("/clear", methods=["POST"])
def clear_history():
    """Clear the chat history."""
    global chat_history
    chat_history = []
    return render_template("chat.html", chat_history=chat_history)

@app.route("/upload", methods=["GET", "POST"])
def upload_report():
    """Handle medical report upload and analysis."""
    if request.method == "POST":
        if 'report' not in request.files:
            return "No file part"
        file = request.files['report']
        if file.filename == '':
            return "No selected file"
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            analysis_result = analyze_report(file_path)
            return render_template("report_analysis.html", result=analysis_result)
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
