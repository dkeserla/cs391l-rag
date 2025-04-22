import os
from .chunking import chunk_document
from .embeddings import get_embedding_model
from .metadata import add_metadata_tags

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader



def build_retriever(
    file_path="data/",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    chunk_strategy="recursive",
    persist_dir="db",
    rebuild=False,
    file_content_types = None,
    webpage_urls=["https://utcs-ml-course.github.io/", "https://utcs-ml-course.github.io/info/schedule/" "https://utcs-ml-course.github.io/info/class_policy/"],
    webpage_content_types=["syllabus", "schedule", "class policies"],
):
    embedder = get_embedding_model(model_name)

    if os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
        if rebuild:
            import shutil
            print(f"[build_retriever] Deleting existing vectorstore at '{persist_dir}'")
            shutil.rmtree(persist_dir)
        else:
            print("[build_retriever] Loading existing Chroma index...")
            return Chroma(persist_directory=persist_dir, embedding_function=embedder).as_retriever()

    print("[build_retriever] No index found or --rebuild flag used — building fresh ChromaDB...")

    docs = []
    if file_content_types is None:
        file_content_types = {}
        

    doc_content_types = []
    for filename in os.listdir(file_path):
        full_path = os.path.join(file_path, filename)
        if filename.endswith(".txt"):
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            docs.append(Document(page_content=content, metadata={"source": filename}))
            doc_content_types.append(file_content_types.get(filename, "transcript"))

        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(full_path)
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = filename
                docs.append(page)
                doc_content_types.append(file_content_types.get(filename, "transcript"))

        print(f"[load] {filename} → content type: {file_content_types.get(filename, 'transcript')}")
    
    print(f"[build_retriever] Loading {len(webpage_urls)} webpages...")
    for url, webpage_content_type in zip(webpage_urls, webpage_content_types):
        try:
            loader = WebBaseLoader(url)
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = url
                docs.append(page)
                doc_content_types.append(webpage_content_type)
        except Exception as e:
            print(f"⚠️ Failed to load {url}: {e}")

    
    docs = add_metadata_tags(docs, content_types=doc_content_types)
    chunks = chunk_document(docs, strategy=chunk_strategy)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=persist_dir
    )

    return vectorstore.as_retriever()
