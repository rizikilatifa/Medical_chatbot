from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

if not PINECONE_API_KEY:
   raise ValueError("PINECONE_API_KEY is missing in environment variables.")

extracted_data = load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunk = text_split(filter_data)

embeddings = download_embeddings()

pinecone_api_key = PINECONE_API_KEY

pc= Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"

existing_indexes = pc.list_indexes().names()
if index_name not in existing_indexes:
   pc.create_index(
      name=index_name,
      dimension=384,  # Dimension of all-MiniLM-L6-v2 embeddings
      metric="cosine",
      spec=ServerlessSpec(cloud="aws", region="us-east-1")
   )

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
   documents=text_chunk,
   index_name=index_name,
   embedding=embeddings
)

