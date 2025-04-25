from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)

def chunk_document(docs, strategy="recursive", chunk_size=2000, chunk_overlap=400):
    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy == "sliding":
        splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ValueError("Unknown chunking strategy")
    return splitter.split_documents(docs)
