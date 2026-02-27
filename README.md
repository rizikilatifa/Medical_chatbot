# Medical Chatbot

A conversational AI assistant for medical and health-related questions, powered by Groq's Llama 3.3 70B model with RAG (Retrieval-Augmented Generation) capabilities.

## Overview

This chatbot helps users get answers to health and medical questions using a modern LLM combined with a medical knowledge base. It retrieves relevant documents from Pinecone vector database and uses them as context for accurate, informed responses.

## Features

- **RAG (Retrieval-Augmented Generation)** - Retrieves relevant medical documents from Pinecone
- **Conversation History** - Remembers context across messages
- **Fast Responses** - Uses Groq API (Llama 3.3 70B) for quick inference
- **Clean Web UI** - Responsive Bootstrap interface
- **New Chat Button** - Clear conversation and start fresh
- **Graceful Fallback** - Works even if RAG fails

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python, Flask |
| **LLM** | Groq API (Llama 3.3 70B) |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **Vector DB** | Pinecone |
| **Frontend** | HTML, CSS, Bootstrap, jQuery |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Get Embedding                                 │
│         (Sentence Transformers - all-MiniLM-L6-v2)              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Query Pinecone                                 │
│              (Find relevant documents)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Build Context                                   │
│         (Format retrieved documents for LLM)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               Call Groq API with Context                         │
│           (Llama 3.3 70B generates response)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Return Answer                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
Medical_chatbot/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Procfile              # For production deployment
├── runtime.txt           # Python version specification
├── .env                  # Environment variables (not committed)
├── templates/
│   └── chat.html         # Chat interface
├── static/
│   └── style.css         # Styling
└── src/
    ├── rag.py            # RAG module (embeddings + Pinecone)
    ├── helper.py         # Utility functions
    └── prompt.py         # System prompts
```

## Setup

### Prerequisites

- Python 3.11+
- A Groq API key (free at [console.groq.com](https://console.groq.com))
- A Pinecone API key (free at [pinecone.io](https://pinecone.io))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rizikilatifa/Medical_chatbot.git
   cd Medical_chatbot
   ```

2. **Create virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   SECRET_KEY=your_random_secret_key_here
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open in browser**
   Navigate to `http://localhost:5000`

### Note on First Query

The first query will take **30-60 seconds** as the embedding model loads into memory. Subsequent queries are fast.

## API Keys

| Service | Purpose | Get it from |
|---------|---------|-------------|
| Groq | LLM inference | [console.groq.com](https://console.groq.com) |
| Pinecone | Vector database | [pinecone.io](https://pinecone.io) |

## Deployment

This app is ready for deployment on platforms like:

- **Render** (recommended, free tier available)
- **Railway** ($5 free credit/month)
- **Fly.io** (free allowance)
- **PythonAnywhere** (free tier)

### Deploy to Render

1. Push code to GitHub
2. Create account on [render.com](https://render.com)
3. Create new Web Service → Connect GitHub repo
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `gunicorn app:app`
6. Add environment variables: `GROQ_API_KEY`, `PINECONE_API_KEY`, `SECRET_KEY`
7. Deploy!

## Development History

### What Was Built

1. **Initial Setup** - Flask app with basic chat interface
2. **Ollama Integration** - Initially used local Ollama with qwen2.5:3b model
3. **Groq Migration** - Switched from Ollama to Groq API for faster responses
4. **Conversation History** - Added session-based conversation memory
5. **RAG Implementation** - Added Pinecone vector search for context retrieval
6. **Production Ready** - Added Procfile, runtime.txt, cleaned up dependencies

### Why Groq Instead of Ollama?

- **Faster responses** - Groq is optimized for speed
- **No local GPU needed** - Runs in the cloud
- **Better model** - Llama 3.3 70B is more capable than local small models
- **Reliable** - No issues with local server hanging

## Future Improvements

- [x] ~~RAG (Retrieval-Augmented Generation) with Pinecone~~ ✅ Done
- [x] ~~Conversation history~~ ✅ Done
- [ ] Multiple language support
- [ ] Voice input/output
- [ ] Mobile app version
- [ ] Stream responses for better UX

## Disclaimer

This chatbot is for **informational purposes only**. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.

## License

MIT License

## Author

Latifa Riziki

---

*Built with assistance from Claude (Anthropic)*
