from flask import Flask, render_template, request, session
from dotenv import load_dotenv
import os
import requests

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Secret key required for Flask sessions
app.secret_key = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# RAG settings
USE_RAG = True  # Set to False to disable RAG
RAG_TOP_K = 3   # Number of documents to retrieve

# System prompt for non-RAG mode
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

    Returns:
        The assistant's response text
    """
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


def call_groq_with_context(query: str, context: str, history: list) -> str:
    """
    Call Groq API with RAG context and conversation history.

    Args:
        query: The current user question
        context: Retrieved context from Pinecone
        history: Conversation history

    Returns:
        The assistant's response text
    """
    # System prompt with RAG context
    system_prompt = f"""You are a helpful medical assistant with access to a medical knowledge base.
Use the following context to answer the user's question. If the answer is not in the context,
provide general medical information but mention that it's not from the knowledge base.

Always remind users that you are an AI and they should consult a real doctor for medical advice.

RETRIEVED CONTEXT FROM KNOWLEDGE BASE:
{context}
"""

    full_messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": query}]

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
    """Get conversation history from session."""
    return session.get("history", [])


def save_history(history: list):
    """Save conversation history to session."""
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    session["history"] = history


def clear_history():
    """Clear the conversation history."""
    session["history"] = []


def get_rag_context(query: str) -> tuple:
    """
    Get RAG context for a query.

    Returns:
        Tuple of (context_string, sources_list) or (None, []) if RAG fails
    """
    if not USE_RAG:
        return None, []

    try:
        from src.rag import retrieve_context
        return retrieve_context(query, top_k=RAG_TOP_K)
    except Exception as e:
        print(f"RAG retrieval failed: {e}", flush=True)
        return None, []


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

        # Try to get RAG context
        context, sources = get_rag_context(msg)

        if context:
            # RAG mode - use context
            print(f"RAG: Found {len(sources)} relevant documents", flush=True)
            answer = call_groq_with_context(msg, context, history)
        else:
            # Fallback mode - regular chat
            print("RAG: No context found, using regular chat", flush=True)
            history.append({"role": "user", "content": msg})
            answer = call_groq(history)

        # Update history
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": answer})
        save_history(history)

        print(f"Bot: {answer[:100]}...", flush=True)

        return answer

    except Exception as e:
        print(f"Error: {e}", flush=True)
        return f"Sorry, an error occurred: {str(e)}"


@app.route("/clear", methods=["POST"])
def clear():
    """Clear conversation history and start fresh."""
    clear_history()
    return "Conversation cleared. Start a new conversation!"


@app.route("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "rag_enabled": USE_RAG}


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Medical Chatbot on port {port}", flush=True)
    print(f"RAG enabled: {USE_RAG}", flush=True)
    print("Conversation history enabled!", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
