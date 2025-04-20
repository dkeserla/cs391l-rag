import os
import time
from dotenv import load_dotenv

from retrieval.base import build_retriever
from retrieval.embeddings import get_embedding_model
from generation.interface import generate_answer
from augmentation.base import augment_context

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def timed_section(label, func, *args, **kwargs):
    print(f"[{label}] Starting...")
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    print(f"[{label}] Completed in {end - start:.2f} seconds.\n")
    return result


def main():
    load_dotenv()

    query = "What is the main topic of Lecture 1?"
    model_name = "gpt-4o"  # "gpt-4o", "gemini", or "claude-3.7"
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # 1. Build retriever
    retriever = timed_section("1/4: Build Retriever", build_retriever, "data/", model_name=embed_model_name)

    # 2. Retrieve documents
    retrieved_docs = timed_section("2/4: Retrieve Documents", retriever.invoke, query)
    print(f"Retrieved {len(retrieved_docs)} documents.")

    # 3. Rerank + compress
    embed_model = timed_section("3/4a: Load Embedder", get_embedding_model, embed_model_name)
    context = timed_section("3/4b: Augment Context", augment_context, query, retrieved_docs, embed_model)

    # 4. Generate answer
    answer = timed_section("4/4: Generate Answer", generate_answer, query, context, model_name=model_name)

    print("Final Answer:\n", answer)


if __name__ == "__main__":
    main()
