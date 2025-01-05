import os
from pdfminer.high_level import extract_text
import pytesseract
from google.generativeai import GenerativeModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Generative AI model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

model = GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

def extract_report_text(filepath):
    """Extract text from a report file (PDF, image, or text)."""
    if filepath.endswith('.pdf'):
        return extract_text(filepath)  # Extract text from PDF
    elif filepath.endswith(('.png', '.jpg', '.jpeg')):
        return pytesseract.image_to_string(filepath)  # Extract text from image
    else:
        with open(filepath, 'r') as file:
            return file.read()  # Read plain text file

def analyze_report(report_text):
    """Simplify and analyze the medical report using AI."""
    try:
        # Start a chat session
        chat_session = model.start_chat(history=[])
        
        # Request analysis from AI model
        response = chat_session.send_message(f"Please simplify this medical report: {report_text}")
        return response.text.strip()
    except Exception as e:
        return f"Error analyzing report: {str(e)}"
