import glob
import logging
import os
import time
from typing import List
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import config

# Configuration
PDF_DIRECTORY = config.PDF_DIRECTORY
FAISS_DB_FILE = config.FAISS_DB_FILE
LOG_FILE = config.LOG_FILE

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)  # Get a logger for this module


def load_pdfs(pdf_directory: str) -> list[Document]:
    filenames = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=48)
    print(f"Loading {len(filenames)} PDFs...")
    for filename in filenames:
        try:
            loader = PyPDFLoader(filename)
            documents.extend(loader.load_and_split(text_splitter=splitter))
            print(f"Loaded {filename}")
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
    return documents


def create_and_persist_embeddings(documents: List[str]) -> FAISS:
    print("Creating embeddings...")
    embeddings = GPT4AllEmbeddings()
    print("Creating vector store...")
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(FAISS_DB_FILE)
    return vector_store


def load_or_create_vector_store() -> FAISS:
    print("Loading vector DB...")
    if os.path.exists(FAISS_DB_FILE):
        print("Loading vector store...")
        embeddings = GPT4AllEmbeddings()
        return FAISS.load_local(FAISS_DB_FILE, embeddings)
    else:
        print("Creating vector store...")
        documents = load_pdfs(PDF_DIRECTORY)
        return create_and_persist_embeddings(documents)


def main():
    llm = Ollama(base_url=config.BASE_URL, model=config.OLLAMA_MODEL)
    vector_store = load_or_create_vector_store()
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_store.as_retriever())
    while True:
        try:
            query = input("\033[94mWhat's your question? (Type 'exit' to quit)\033[0m\n")
            if query == "exit":
                break
            start_time = time.time()
            result = qa_chain.invoke({"query": query})["result"]
            end_time = time.time()
            print(f"\033[92m{result}\033[0m (Response time: {end_time - start_time:.2f}s)")
        except Exception as e:
            logger.exception("An error occurred during query processing.")
            print("An error occurred. Please try again.")


if __name__ == '__main__':
    print("Starting main...")
    main()
