from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

PERSIST_DIR = "vectordb"
COLLECTION_NAME = "chest_xray_knowledge"

embeddings = OllamaEmbeddings(
    model="embeddinggemma:latest"
)

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)