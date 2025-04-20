import os
from .chunking import chunk_document
from .embeddings import get_embedding_model
from .metadata import add_metadata_tags

from langchain_chroma import Chroma  # ✅ new import
from langchain_core.documents import Document


def build_retriever(
    file_path="data/",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    chunk_strategy="recursive",
    persist_dir="db"
):
    embedder = get_embedding_model(model_name)

    # Use existing persisted vectorstore if it exists
    if os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
        print("[build_retriever] Loading existing Chroma index...")
        return Chroma(persist_directory=persist_dir, embedding_function=embedder).as_retriever()

    # No vectorstore found, build from documents
    print("[build_retriever] No index found — building fresh ChromaDB...")

    docs = []
    for filename in os.listdir(file_path):
        if filename.endswith(".txt"):
            full_path = os.path.join(file_path, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            docs.append(Document(page_content=content, metadata={"source": filename}))

    docs = add_metadata_tags(docs, content_type="transcript")
    chunks = chunk_document(docs, strategy=chunk_strategy)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=persist_dir
    )

    return vectorstore.as_retriever()
