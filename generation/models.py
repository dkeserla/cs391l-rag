import os

# OpenAI
from langchain_openai import ChatOpenAI

# Gemini
import google.generativeai as genai

# Claude
import anthropic


def build_prompt(query: str, context: str) -> str:
    """
    Consistent system prompt used across all LLMs.
    """
    system_instruction = (
        "You are a helpful teaching assistant for a graduate-level machine learning course. "
        "Use only the information provided in the context to answer the student's question. "
    )

    return f"""{system_instruction}\n\nContext:\n{context}\n\nQuestion:\n{query}""".strip()


def get_openai_response(query: str, context: str) -> str:
    """
    Uses GPT-4o to generate a response based on query and context.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = build_prompt(query, context)
    response = llm.invoke(prompt)
    return response.content


def get_gemini_response(query: str, context: str) -> str:
    """
    Uses Gemini 2 to generate a response based on query and context.
    """
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = build_prompt(query, context)
    response = model.generate_content(prompt)
    return response.text.strip()


def get_claude_response(query: str, context: str) -> str:
    """
    Uses Claude 3.7 (or Claude 3 Opus) to generate a response based on query and context.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    prompt = build_prompt(query, context)

    response = client.messages.create(
        model="claude-3-opus-20240229",  # Change to "claude-3.7" if available
        max_tokens=1000,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text.strip()
