import requests
from typing import List, Dict, Any

class SimpleGroqLLM:
    """Simple Groq LLM using direct API calls - no langchain dependencies."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", temperature: float = 0):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    def invoke(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Call the Groq API and return the response."""
        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature
            },
            timeout=60
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return {"content": content}
