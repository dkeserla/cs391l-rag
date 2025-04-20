from langchain_community.vectorstores import Chroma

def build_hybrid_retriever(chunks, embedding_model, persist_dir="db"):
    # For now, use Chroma as backend — future: combine with BM25
    vectorstore = Chroma.from_documents(chunks, embedding_model, persist_directory=persist_dir)
    return vectorstore.as_retriever()
