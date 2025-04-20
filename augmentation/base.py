from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer, util

def compress_documents(documents: list[Document], max_chars: int = 3000) -> str:
    """
    Naively truncates concatenated context to a max character length.
    (Can be replaced with summarization or chunk scoring later.)
    """
    combined = ""
    for doc in documents:
        if len(combined) + len(doc.page_content) > max_chars:
            break
        combined += doc.page_content + "\n"
    return combined.strip()


from langchain_core.documents import Document
import torch
from sentence_transformers import util

def rerank_documents(query: str, documents: list[Document], embedder, top_k=5) -> list[Document]:
    """
    Works with any LangChain Embeddings object. Converts output to PyTorch for cosine similarity.
    """
    query_embedding = torch.tensor(embedder.embed_query(query)).unsqueeze(0)
    doc_embeddings = torch.tensor(embedder.embed_documents([doc.page_content for doc in documents]))

    similarities = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
    doc_sim_pairs = list(zip(documents, similarities))
    doc_sim_pairs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in doc_sim_pairs[:top_k]]


def augment_context(query: str, retrieved_docs: list[Document], embedder=SentenceTransformer("all-MiniLM-L6-v2")) -> str:
    """
    Applies reranking and context compression to prepare final context string.
    """
    reranked = rerank_documents(query, retrieved_docs, embedder)
    compressed = compress_documents(reranked)
    return compressed
