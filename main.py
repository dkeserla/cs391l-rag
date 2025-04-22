import os
import time
from dotenv import load_dotenv

from retrieval.base import build_retriever
from retrieval.embeddings import get_embedding_model
from generation.interface import generate_answer
from augmentation.base import augment_context
import argparse

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
   
    parser = argparse.ArgumentParser(description="Run a RAG query on lecture transcripts.")
    parser.add_argument("--query", type=str, help="The question to ask.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM to use: gpt-4o, gemini, claude-3.7")
    parser.add_argument("--embed", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of the vectorstore from documents")

    args = parser.parse_args()

    query = args.query or "What percentage of the grade is homework"
    model_name = args.model or "gpt-4o"  # "gpt-4o", "gemini", or "claude-3.7"
    embed_model_name = args.embed or "sentence-transformers/all-MiniLM-L6-v2"

    # 1. Build retriever
    retriever = timed_section("1/4: Build Retriever", build_retriever, "data/", model_name=embed_model_name, file_content_types={"hw1.pdf": "homework 1", "hw2.pdf": "homework 2", "hw3.pdf": "homework 3","hw4.pdf": "homework 4", "merged_transcript.txt": "lecture transcript"}, rebuild=args.rebuild)

    # 2. Retrieve documents
    retrieved_docs = timed_section("2/4: Retrieve Documents", retriever.invoke, query)
    print(f"Retrieved {len(retrieved_docs)} documents.")

    # 3. Rerank + compress
    embed_model = timed_section("3/4a: Load Embedder", get_embedding_model, embed_model_name)
    context = timed_section("3/4b: Augment Context", augment_context, query, retrieved_docs, embed_model)

    # 4. Generate answer
    answer = timed_section("4/4: Generate Answer", generate_answer, query, context, model_name=model_name)

    print(f'Query: {query}')
    print(f'Answer: {answer}')


if __name__ == "__main__":
    main()
