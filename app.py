from flask import Flask, render_template, request
import os
import requests

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def call_groq(prompt: str) -> str:
    """Call Groq API directly."""
    system_prompt = """You are a helpful medical assistant. Answer the user's questions about health and medicine.
Always remind users that you are an AI and they should consult a real doctor for medical advice."""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0
        },
        timeout=60
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form['msg']
        print(f"User: {msg}", flush=True)

        # Call Groq directly
        answer = call_groq(msg)
        print(f"Bot: {answer}", flush=True)

        return answer
    except Exception as e:
        print(f"Error: {e}", flush=True)
        return f"Sorry, an error occurred: {str(e)}"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Medical Chatbot on port {port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
