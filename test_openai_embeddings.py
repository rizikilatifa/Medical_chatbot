import sys
import os
print("=== Testing OpenAI Embeddings ===", flush=True)

from dotenv import load_dotenv
load_dotenv()

print(f"OPENAI_API_KEY set: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}", flush=True)

print("Importing OpenAIEmbeddings...", flush=True)
from langchain_openai import OpenAIEmbeddings

print("Creating embeddings instance...", flush=True)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

print("Testing embedding...", flush=True)
test = embeddings.embed_query("test query")
print(f"Embedding dimension: {len(test)}", flush=True)
print("SUCCESS!", flush=True)
