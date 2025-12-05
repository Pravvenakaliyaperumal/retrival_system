import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# NEW LangChain paths
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Gemini imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Load environment vars
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# -------------------------------
# PDF Extractor
# -------------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


# -------------------------------
# Split into chunks
# -------------------------------
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_text(text)


# -------------------------------
# VECTOR DB using Gemini Embeddings
# -------------------------------
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",         # 💥 THIS WORKS (PaLM is dead)
        google_api_key=GOOGLE_API_KEY
    )

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


# -------------------------------
# Conversational RAG Chain
# -------------------------------
def get_conversational_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",                        # or "gemini-1.5-flash"
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    return conversation_chain
