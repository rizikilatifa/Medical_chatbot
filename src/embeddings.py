import requests
from typing import List
from langchain_core.embeddings import Embeddings

class FastOpenAIEmbeddings(Embeddings):
    """Fast OpenAI embeddings using direct API calls - no torch dependency."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1/embeddings"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "input": texts
            }
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]

    def embed_query(self, text: str) -> List[float]:
        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "input": text
            }
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
