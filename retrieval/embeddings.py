from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from transformers import CLIPTokenizer, CLIPTextModel
import torch

# === 1. LangChain-compatible CLIP wrapper ===

class CLIPEmbedding(Embeddings):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)

    def embed_documents(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :].tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]

# === 2. Returns LangChain-compatible Embeddings object ===

def get_embedding_model(model_name: str) -> Embeddings:
    if "sentence-transformers" in model_name:
        return HuggingFaceEmbeddings(model_name=model_name)
    elif "gemini" in model_name:
        raise NotImplementedError("Gemini embedding not yet integrated")
    elif "clip" in model_name:
        return CLIPEmbedding()
    else:
        raise ValueError(f"Unsupported embedding model: {model_name}")
