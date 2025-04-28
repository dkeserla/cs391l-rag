from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter

def compress_documents(query: str,
                       documents: list[Document],
                       embedder,
                       max_chars: int = 3000,
                       chunk_size: int = 512,
                       chunk_overlap: int = 100) -> str:
    """
    Naively truncates concatenated context to a max character length.
    (Can be replaced with summarization or chunk scoring later.)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []
    for doc in documents:
        chunks.extend(splitter.split_text(doc.page_content))

    q_emb = torch.tensor(embedder.embed_query(query)).unsqueeze(0)
    c_emb = torch.tensor(embedder.embed_documents(chunks))
    sims = util.pytorch_cos_sim(q_emb, c_emb)[0]

    ranked = sorted(zip(chunks, sims.tolist()), key=lambda x : x[1], reverse=True)

    context, used = [], 0
    for text, _ in ranked:
        if used + len(text) + 1 > max_chars:
            break
        context.append(text)
        used += len(text) + 1

    return "\n".join(context).strip()


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
    Applies reranking and context compression to prepare final context string. TODO
    """
    reranked = rerank_documents(query, retrieved_docs, embedder)

    context = compress_documents(query, reranked, embedder)

    return context
