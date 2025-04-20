from generation.models import (
    get_openai_response,
    get_gemini_response,
    get_claude_response
)

SUPPORTED_MODELS = ["gpt-4o", "gemini-2", "claude-3.7"]

def generate_answer(query: str, context: str, model_name: str = "gpt-4o") -> str:
    if model_name == "gpt-4o":
        return get_openai_response(query, context)
    elif model_name == "gemini-2":
        return get_gemini_response(query, context)
    elif model_name == "claude-3.7":
        return get_claude_response(query, context)
    else:
        raise ValueError(f"Model '{model_name}' not supported. Choose from {SUPPORTED_MODELS}.")
