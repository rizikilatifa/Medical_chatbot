from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.embeddings import FastOpenAIEmbeddings
from typing import List
from langchain_core.documents import Document
import os

# Extract Data from the PDF File
def load_pdf_file(data):
   loader = DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)


   documents = loader.load()

   return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
   """
   Given a list of Document objects, return a new list of Document objects
   containing only 'source in metadata and the original page_content.

   """
   minimal_docs: List[Document] = []
   for doc in docs:
      src = doc.metadata.get("source")
      minimal_docs.append(
         Document(
            page_content=doc.page_content,
            metadata={"source": src}
         )
      )
   return minimal_docs


def text_split(minimal_docs):
   text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 500,
      chunk_overlap= 20,
   )
   texts_chunk = text_splitter.split_documents(minimal_docs)
   return texts_chunk


def download_embeddings():
   """
   Return fast OpenAI embeddings (no torch dependency).
   """
   embeddings = FastOpenAIEmbeddings(
      api_key=os.getenv("OPENAI_API_KEY"),
      model="text-embedding-3-small"
   )
   return embeddings
