from flask import Flask, render_template, request, session
from dotenv import load_dotenv
import os
import requests

load_dotenv()

app = Flask(__name__)

# Secret key required for Flask sessions
app.secret_key = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# System prompt for the medical assistant
SYSTEM_PROMPT = """You are a helpful medical assistant. Answer the user's questions about health and medicine.
Always remind users that you are an AI and they should consult a real doctor for medical advice.
Remember the context of our conversation and refer back to previous messages when relevant."""

# Maximum messages to keep in history (to avoid token limits)
MAX_HISTORY = 20

def call_groq(messages: list) -> str:
    """
    Call Groq API with full conversation history.

    Args:
        messages: List of message dicts with 'role' and 'content'
                  e.g., [{"role": "user", "content": "Hello"}]

    Returns:
        The assistant's response text
    """
    # Prepend system prompt to messages
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": full_messages,
            "temperature": 0
        },
        timeout=60
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def get_history() -> list:
    """Get conversation history from session, or return empty list."""
    return session.get("history", [])

def save_history(history: list):
    """Save conversation history to session, trimming if too long."""
    # Keep only last MAX_HISTORY messages to avoid token limits
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    session["history"] = history

def clear_history():
    """Clear the conversation history."""
    session["history"] = []

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form['msg']
        print(f"User: {msg}", flush=True)

        # Get existing conversation history
        history = get_history()

        # Add user message to history
        history.append({"role": "user", "content": msg})

        # Call Groq with full history
        answer = call_groq(history)

        # Add bot response to history
        history.append({"role": "assistant", "content": answer})

        # Save updated history
        save_history(history)

        print(f"Bot: {answer}", flush=True)

        return answer
    except Exception as e:
        print(f"Error: {e}", flush=True)
        return f"Sorry, an error occurred: {str(e)}"

@app.route("/clear", methods=["POST"])
def clear():
    """Clear conversation history and start fresh."""
    clear_history()
    return "Conversation cleared. Start a new conversation!"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Medical Chatbot on port {port}", flush=True)
    print("Conversation history enabled!", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
