import os
import streamlit as st
from ingest import ingest_pdf
from rag import llm, embeddings
from langchain_chroma import Chroma

st.set_page_config(
    page_title="RAG Chat App",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Streamlit RAG Chat App")

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

    db = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your PDF")

    if question:

        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        docs = retriever.invoke(question)

st.write("Retrieved documents:", len(docs))

context = "\n\n".join(
    doc.page_content for doc in docs
)

prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the context below.

Context:
{context}

Question:
{question}
"""

st.text_area("Prompt sent to LLM", prompt, height=300)

response = llm.invoke(prompt)
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
    response = st.write("Retrieved Documents:", len(docs))

for i, doc in enumerate(docs):
    st.write(f"Document {i+1}")
    st.write(doc.page_content[:300])llm.invoke(prompt)
    answer = response.content
except Exception as e:
    answer = f"Error: {e}"

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )