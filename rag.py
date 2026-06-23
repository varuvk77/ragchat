from langchain_ollama import ChatOllama, OllamaEmbeddings

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)
