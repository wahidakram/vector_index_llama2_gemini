import glob
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

import config

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

filenames = glob.glob(os.path.join(config.PDF_DIRECTORY, "*.pdf"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question concisely, focusing on the most relevant and important details from the PDF context. 
    Refrain from mentioning any mathematical equations, even if they are present in provided context. 
    Focus on the textual information available. Please provide direct quotations or references from PDF
    to back up your response. If the answer is not found within the PDF, 
    please state "answer is not available in the context."\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Example response format:
    Overview: 
    (brief summary or introduction)
    Key points: 
    (point 1: paragraph for key details)
    (point 2: paragraph for key details)
    ...
    Use a mix of paragraphs and points to effectively convey the information.
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)

    return response


def main():
    if not os.path.exists("faiss_index"):
        pdf_text = get_pdf_text(filenames)
        print("Total PDF Text Length:", len(pdf_text))
        text_chunks = get_text_chunks(pdf_text)
        print("Total Text Chunks:", len(text_chunks))
        get_vector_store(text_chunks)
    while True:
        try:
            user_question = input("What's your question? (Type 'exit' to quit)\n")
            if user_question == "exit":
                break
            print(user_input(user_question))
        except Exception as e:
            print("An error occurred. Please try again.")


if __name__ == "__main__":
    main()
