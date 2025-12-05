import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain


def user_input(user_question):
    response = st.session_state.conversation.run(input=user_question)
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.markdown(f"**User:** {message.content}")
        else:
            st.markdown(f"**AI:** {message.content}")
 

def main():
    st.set_page_config(page_title="information retrival", layout="centered")
    st.header("Information Retrieval System")

    user_question = st.text_input("Ask your question about the pdf files here:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs= st.file_uploader("upload your pdf files and click on the submit & Process Button", accept_multiple_files=True)
        if st.button("submit & Process"):
            with st.spinner("Processing"):
                raw_text= get_pdf_text(pdf_docs)
                text_chunks= get_text_chunks(raw_text)
                vector_store= get_vector_store(text_chunks)
                st.session_state.conversation= get_conversational_chain(vector_store)

                




                st.success("your pdf files are processed successfully")

if __name__ == "__main__":
    main()