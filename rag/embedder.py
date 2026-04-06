# rag/embedder.py
# Loads the sentence-transformer model and converts text into vectors

from sentence_transformers import SentenceTransformer

# Using a lightweight but powerful model (downloads once, ~90MB)
MODEL_NAME = "all-MiniLM-L6-v2"

# Load model once at module level (avoids reloading every call)
model = SentenceTransformer(MODEL_NAME)


def embed_text(text: str):
    """
    Convert a single string into a numpy embedding vector.
    Returns: numpy array of shape (384,)
    """
    return model.encode(text, convert_to_numpy=True)


def embed_texts(texts: list):
    """
    Convert a list of strings into embedding vectors.
    Returns: numpy array of shape (N, 384)
    """
    return model.encode(texts, convert_to_numpy=True)