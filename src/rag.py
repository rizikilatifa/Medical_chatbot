"""
RAG (Retrieval-Augmented Generation) module for Medical Chatbot.

This module handles:
1. Creating embeddings for user queries (using sentence-transformers)
2. Querying Pinecone vector database
3. Building context from retrieved documents
"""

import os
import requests
from typing import List, Optional

# API Keys (loaded at module import time)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = "medical-chatbot"

# Pinecone host URL (obtained from Pinecone dashboard)
PINECONE_HOST = "https://medical-chatbot-x0js8of.svc.aped-4627-b74a.pinecone.io"

# Embedding model (cached after first load)
_embedding_model = None


def get_embedding_model():
    """
    Get or create the sentence-transformers embedding model.

    This is loaded lazily and cached to avoid slow startup times.
    The model is loaded on first query, not at app startup.
    """
    global _embedding_model

    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading embedding model (first query may take 30-60 seconds)...", flush=True)
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Embedding model loaded successfully!", flush=True)
        except ImportError:
            raise Exception("sentence-transformers not installed. Run: pip install sentence-transformers")
        except Exception as e:
            raise Exception(f"Failed to load embedding model: {e}")

    return _embedding_model


def get_embedding(text: str) -> List[float]:
    """
    Get embedding vector for text using sentence-transformers.

    Uses the same model (all-MiniLM-L6-v2) that was used to create
    the Pinecone index, ensuring dimension compatibility (384 dims).

    Args:
        text: The text to embed

    Returns:
        List of floats representing the embedding vector
    """
    model = get_embedding_model()
    embedding = model.encode(text)
    return embedding.tolist()


def query_pinecone(query_embedding: List[float], top_k: int = 3) -> List[dict]:
    """
    Query Pinecone vector database for similar documents.

    Args:
        query_embedding: The embedding vector of the query
        top_k: Number of documents to retrieve

    Returns:
        List of matching documents with their metadata and scores
    """
    if not PINECONE_API_KEY:
        raise Exception("PINECONE_API_KEY not set in environment")

    response = requests.post(
        f"{PINECONE_HOST}/query",
        headers={
            "Api-Key": PINECONE_API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json().get("matches", [])


def build_context(matches: List[dict]) -> str:
    """
    Build context string from retrieved documents.

    Args:
        matches: List of document matches from Pinecone

    Returns:
        Formatted context string for the LLM
    """
    if not matches:
        return "No relevant documents found in the knowledge base."

    context_parts = []
    for i, match in enumerate(matches, 1):
        text = match.get("metadata", {}).get("text", "")
        score = match.get("score", 0)

        if text:
            context_parts.append(f"[Document {i}] (Relevance: {score:.2f})\n{text}\n")

    return "\n".join(context_parts) if context_parts else "No relevant documents found."


def retrieve_context(query: str, top_k: int = 3) -> tuple:
    """
    Main RAG retrieval function.

    Args:
        query: The user's question
        top_k: Number of documents to retrieve

    Returns:
        Tuple of (context_string, list_of_sources)
        Returns (None, []) if retrieval fails
    """
    try:
        # Step 1: Get embedding for query
        embedding = get_embedding(query)

        # Step 2: Query Pinecone
        matches = query_pinecone(embedding, top_k)

        # Step 3: Build context
        context = build_context(matches)

        # Get sources for citation
        sources = [m.get("metadata", {}).get("source", "Unknown") for m in matches]

        return context, sources

    except Exception as e:
        print(f"RAG Error: {e}", flush=True)
        return None, []
