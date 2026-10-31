
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings


# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "travel_policy"
EMBEDDING_MODEL = "nomic-embed-text"


def ingest_documents():
    documents = []

    # Read the three policy documents
    filenames = [
        "travel_policy.txt",
        "hotel_limits.txt",
        "claim_procedure.txt",
    ]

    for filename in filenames:
        file_path = DATA_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Policy document not found: {file_path}"
            )

        text = file_path.read_text(encoding="utf-8")

        if not text.strip():
            raise ValueError(
                f"Policy document is empty: {filename}"
            )

        documents.append(
            Document(
                page_content=text,
                metadata={"source": filename},
            )
        )

    # Split documents into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )

    chunks = splitter.split_documents(documents)

    # Create local embeddings using Ollama
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    # Create or open the persistent ChromaDB collection
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # Clear previous chunks to prevent duplicates
    vector_store.reset_collection()

    # Store the new chunks
    vector_store.add_documents(chunks)

    print("Document ingestion completed successfully!")
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks stored: {len(chunks)}")
    print(f"ChromaDB location: {CHROMA_DIR}")


if __name__ == "__main__":
    ingest_documents()
