import sys
import os

log_file = open("startup_log.txt", "w")

def log(msg):
    print(msg, flush=True)
    log_file.write(msg + "\n")
    log_file.flush()

log("=== Starting Medical Chatbot ===")

log("1. Loading environment...")
from dotenv import load_dotenv
load_dotenv()
log("   Environment loaded")

log("2. Loading embeddings...")
from src.helper import download_embeddings
embeddings = download_embeddings()
log("   Embeddings loaded")

log("3. Connecting to Pinecone...")
from langchain_pinecone import PineconeVectorStore
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
log("   Pinecone connected")

log("4. Setting up LLM...")
from langchain_openai import ChatOpenAI
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)
log("   LLM ready")

log("5. Creating RAG chain...")
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.prompt import prompt

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
doc_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, doc_chain)
log("   RAG chain created")

log("6. Starting Flask server...")
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def cha():
    msg = request.form['msg']
    log(f"User: {msg}")
    response = rag_chain.invoke({"input": msg})
    log(f"Bot: {response['answer']}")
    return str(response['answer'])

log("=== Server running on http://localhost:8080 ===")
app.run(host="0.0.0.0", port=8080, debug=True)
