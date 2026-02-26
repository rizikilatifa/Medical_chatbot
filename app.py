from flask import Flask, render_template, request
from src.helper import download_embeddings
from src.llm import FastGroqLLM
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from src.prompt import prompt
import os


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_embeddings()

index_name = "medical-chatbot"
# embed each chunk and upsert the embeddings into your pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
   index_name=index_name,
   embedding=embeddings
)



retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = FastGroqLLM(
   model="llama-3.3-70b-versatile",
   temperature=0,
   api_key=GROQ_API_KEY
)

doc_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, doc_chain)

@app.route("/")
def index():
   return render_template('chat.html')

@app.route("/get", methods = ["GET", "POST"])
def cha():
   msg = request.form['msg']
   print(msg)
   response = rag_chain.invoke({"input": msg})
   print("Response : ", response["answer"])
   return str(response['answer'])


if __name__ == '__main__':
   app.run(host="0.0.0.0", port = 5000, debug= True)
