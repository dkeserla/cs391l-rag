from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)

def chunk_document(docs, strategy="recursive"):
    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    elif strategy == "sliding":
        splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=200)
    else:
        raise ValueError("Unknown chunking strategy")
    return splitter.split_documents(docs)
