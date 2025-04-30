import os
import time
from dotenv import load_dotenv

from retrieval.base import build_retriever
from retrieval.embeddings import get_embedding_model
from generation.interface import generate_answer
from augmentation.base import augment_context
import argparse

from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI


metadata_field_info = [
    AttributeInfo(
        name="type",
        description="The type of content in the document. Can be 'homework', 'lecture transcript','syllabus', 'schedule', or 'class policies'.",
        type="string",
    ),
   AttributeInfo(
        name="homework_number",
        description="The homework number for the document. If the document is not a homework assignment, this field will be absent.",
        type="integer",
    ),
]

document_content_description = (
    "The documents contains information about the course, including lecture transcripts, homework assignments, and class policies. "
    "For example "
    "if the query is about homework 2, your filter should be filter=Operation(operator=<Operator.AND: 'and'>, "
    "arguments=[Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='type', value='homework'), Comparison(comparator=<Comparator.EQ: 'eq'>, "
    "attribute='homework_number', value=2)] "

    "if the query is related to the syllabus the filter should be filter=Comparison(comparator=<Comparator.EQ: 'eq'>, "
    "attribute='type', value='syllabus') "
    )

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
    parser.add_argument("--model", type=str, default="gemini-2", help="LLM to use: gpt-4o, gemini-2, claude-3.7")
    parser.add_argument("--embed", type=str, default="openai/text-embedding-3-small", help="Embedding model")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of the vectorstore from documents")
    parser.add_argument("--self_query", action="store_true", help="Use self-query retriever")

    args = parser.parse_args()

    query = args.query or "What percentage of the grade is homework"
    model_name = args.model 
    embed_model_name = args.embed 
    self_query = args.self_query

    

    # 1. Build retriever
    vectorstore = timed_section("1/4: Build Retriever", build_retriever, "data/", model_name=embed_model_name, file_content_types={"hw1.pdf": {"type": "homework", "homework_number": 1} , "hw2.pdf": {"type": "homework", "homework_number": 2}, "hw3.pdf": {"type": "homework", "homework_number": 3}, "hw4.pdf": {"type": "homework", "homework_number": 4}, "merged_transcript.txt": {"type": "lecture transcripts"} }, rebuild=args.rebuild)
    if self_query:
        llm = ChatOpenAI(temperature=0)
        retriever = SelfQueryRetriever.from_llm(
            llm,
            vectorstore,
            document_content_description,
            metadata_field_info,
            search_kwargs={"k":5},
        )
    # Suppose your retriever is called "retriever"
    else:
        retriever = vectorstore.as_retriever()

    # retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 2. Retrieve documents
    retrieved_docs = timed_section("2/4: Retrieve Documents", retriever.invoke, query)
  

    # 3. Rerank + compress
    embed_model = timed_section("3/4a: Load Embedder", get_embedding_model, embed_model_name)
    context = timed_section("3/4b: Augment Context", augment_context, query, retrieved_docs, embed_model)
    print("context:", context)
    # # 4. Generate answer
    answer = timed_section("4/4: Generate Answer", generate_answer, query, context, model_name=model_name)

    print(f'Query: {query}')
    print(f'Answer: {answer}')


if __name__ == "__main__":
    main()
