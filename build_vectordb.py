import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DOCS_PATH = "rag_docs"
PERSIST_DIR = "vectordb"
COLLECTION_NAME = "chest_xray_knowledge"

def build_vector_db():
    documents = []

    for file in os.listdir(DOCS_PATH):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(DOCS_PATH, file))
            documents.extend(loader.load())

    print(f"Loaded {len(documents)} raw documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = OllamaEmbeddings(
        model="embeddinggemma:latest"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )

    vectorstore.persist()
    print("✅ Vector DB successfully built and persisted")

if __name__ == "__main__":
    build_vector_db()
