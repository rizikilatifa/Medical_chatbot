# Medical Chatbot

A conversational AI assistant for medical and health-related questions, powered by Groq's Llama 3.3 70B model.

## Overview

This chatbot helps users get answers to health and medical questions using a modern LLM (Large Language Model). It features a clean web interface built with Flask and uses the Groq API for fast, intelligent responses.

## Features

- Interactive chat interface
- Fast response times using Groq API (Llama 3.3 70B)
- Clean, responsive web UI
- Medical-focused system prompt with appropriate disclaimers

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python, Flask |
| **LLM** | Groq API (Llama 3.3 70B) |
| **Frontend** | HTML, CSS, Bootstrap |
| **Vector DB** | Pinecone (for future RAG integration) |

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
    ├── helper.py         # Utility functions
    ├── prompt.py         # System prompts
    └── embed.py          # Embedding utilities
```

## Setup

### Prerequisites

- Python 3.11+
- A Groq API key (free at [console.groq.com](https://console.groq.com))

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
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open in browser**
   Navigate to `http://localhost:5000`

## API Keys

| Service | Purpose | Get it from |
|---------|---------|-------------|
| Groq | LLM inference | [console.groq.com](https://console.groq.com) |
| Pinecone | Vector database | [pinecone.io](https://pinecone.io) |
| OpenAI | Embeddings | [platform.openai.com](https://platform.openai.com) |

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
6. Add environment variable: `GROQ_API_KEY`
7. Deploy!

## Development History

### What Was Built

1. **Initial Setup** - Flask app with basic chat interface
2. **Ollama Integration** - Initially used local Ollama with qwen2.5:3b model
3. **Groq Migration** - Switched from Ollama to Groq API for faster responses
4. **Embeddings** - Attempted HuggingFace embeddings (slow on Windows) then OpenAI embeddings
5. **Production Ready** - Added Procfile, runtime.txt, cleaned up dependencies

### Why Groq Instead of Ollama?

- **Faster responses** - Groq is optimized for speed
- **No local GPU needed** - Runs in the cloud
- **Better model** - Llama 3.3 70B is more capable than local small models
- **Reliable** - No issues with local server hanging

## Future Improvements

- [ ] RAG (Retrieval-Augmented Generation) with Pinecone
- [ ] Conversation history
- [ ] Multiple language support
- [ ] Voice input/output
- [ ] Mobile app version

## Disclaimer

This chatbot is for **informational purposes only**. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.

## License

MIT License

## Author

Latifa Riziki

---

*Built with assistance from Claude (Anthropic)*
