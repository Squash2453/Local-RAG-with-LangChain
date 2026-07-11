from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from config import EMBEDDING_MODEL,OLLAMA_URL

def build_vectorstore(chunks):
    """Embed all chunks and store them in a FAISS vector store."""

    print(f"    Calculating embeddings for {len(chunks)} chunks using '{EMBEDDING_MODEL}'...")
    print("    This may take a moment...")

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_URL
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    print("    Embeddings complete! Document stored in vector space.")
    return vectorstore