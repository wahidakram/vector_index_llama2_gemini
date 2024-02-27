import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
BASE_URL = os.getenv("BASE_URL", "http://localhost:11434")
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "./pdfs")
FAISS_DB_FILE = os.getenv("FAISS_DB_FILE", "")
LOG_FILE = os.getenv("LOG_FILE", "app.log")