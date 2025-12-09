import streamlit as st
from src.helper import (
    get_pdf_text,
    get_text_chunks,
    get_vector_store,
    get_conversational_chain,
)

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

if "conversation" not in st.session_state:
    st.session_state.conversation = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

st.title("📚 RAG Chatbot Using Gemini + LangChain")
st.write("Upload a PDF and ask questions based on its content.")


# ----------------------------
# SIDEBAR FOR PDF UPLOAD
# ----------------------------
with st.sidebar:
    st.header("Upload PDF documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs", accept_multiple_files=True, type=["pdf"]
    )

    if uploaded_files:
        with st.spinner("Reading PDF and building vector database..."):
            text = get_pdf_text(uploaded_files)
            chunks = get_text_chunks(text)
            vector_store = get_vector_store(chunks)

            st.session_state.vector_store = vector_store
            st.session_state.conversation = get_conversational_chain(vector_store)

        st.success("PDF processed and vector DB created successfully!")


# ----------------------------
# MAIN CHAT INTERFACE
# ----------------------------
user_question = st.text_input("Ask a question about your uploaded documents:")

if user_question:
    if st.session_state.conversation is None:
        st.error("Please upload a PDF first.")
    else:
        with st.spinner("Thinking..."):
            response = st.session_state.conversation.invoke(user_question)

        st.write("### 📘 Answer:")
        st.write(response)
