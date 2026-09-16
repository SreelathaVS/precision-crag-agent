import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_faiss_index(docs: list[str]) -> FAISS:
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    documents = [Document(page_content=text) for text in docs]
    split_docs = splitter.split_documents(documents)
    
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

def get_default_retriever():
    corpus = [
        "FastAPI provides high performance and automated OpenAPI/Swagger documentation.",
        "LangGraph models agent workflows as cyclical state graphs with nodes and conditional edges.",
        "Corrective RAG (CRAG) evaluates retrieval relevance and rewrites queries if context is missing.",
        "LLM Evals leverage assertion testing and LLM-as-a-judge scoring to detect hallucinations."
    ]
    db = build_faiss_index(corpus)
    return db.as_retriever(search_kwargs={"k": 2})
