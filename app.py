import os
import streamlit as st
from langchain_chroma import Chroma

from ingest import ingest_pdf
from rag import llm, embeddings

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="RAG Chat App",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Streamlit RAG Chat App")

# Create uploads folder if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file:

    file_path = os.path.join("uploads", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ PDF uploaded successfully!")

    with st.spinner("Creating embeddings..."):
        ingest_pdf(file_path)

    st.success("✅ PDF indexed successfully!")

    # Load Chroma database
    db = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    question = st.chat_input("Ask a question about your PDF")

    if question:

        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        # Retrieve relevant chunks
        docs = retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the context below.

If the answer is not available in the context, reply:

'I couldn't find that information in the uploaded document.'

Context:
{context}

Question:
{question}
"""

        try:
            response = llm.invoke(prompt)

            if hasattr(response, "content"):
                answer = response.content
            else:
                answer = str(response)

        except Exception as e:
            answer = f"Error: {e}"

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )