import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

print("API KEY LOADED:", GOOGLE_API_KEY)

# ----------------------------
# EXTRACT TEXT FROM PDF
# ----------------------------


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


# ----------------------------
# SPLIT TEXT INTO CHUNKS
# ----------------------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    return splitter.split_text(text)


# ----------------------------
# CREATE VECTOR STORE (FAISS)
# ----------------------------
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY,
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


# ----------------------------
# BUILD RAG PIPELINE (NEW STYLE)
# ----------------------------
def get_conversational_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        """
Use the below retrieved context to answer the question.
Do NOT use outside knowledge. Only answer from the provided context.

Context:
{context}

Question:
{question}
"""
    )

    # LCEL Pipeline: question → retriever → prompt → LLM → output
    rag_chain = (
        {
            "question": RunnablePassthrough(),
            "context": retriever,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
