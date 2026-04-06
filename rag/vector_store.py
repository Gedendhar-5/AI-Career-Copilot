# rag/vector_store.py

import os
import json
import faiss
import numpy as np
from rag.embedder import embed_texts

print("vector_store.py loaded")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_PATH  = os.path.join(BASE_DIR, "..", "data", "faiss_index", "index.faiss")
CHUNKS_PATH = os.path.join(BASE_DIR, "..", "data", "faiss_index", "chunks.json")


def build_index(chunks: list):
    if not chunks:
        return None, []

    texts   = [c["text"] for c in chunks]
    vectors = embed_texts(texts).astype(np.float32)

    dim   = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"index built: {index.ntotal}")
    return index, chunks


def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        return None, []

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"index loaded: {index.ntotal}")
    return index, chunks