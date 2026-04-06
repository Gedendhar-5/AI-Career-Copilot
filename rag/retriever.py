# rag/retriever.py
import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, ".."))

print("retriever.py loaded")

from rag.embedder import embed_text
from rag.vector_store import build_index, load_index
from rag.parser import build_structured_chunks, extract_skills_from_text


class Retriever:
    def __init__(self):
        self.index  = None
        self.chunks = []

    def build_from_texts(self, resume_text: str, jd_text: str):
        structured_chunks = build_structured_chunks(resume_text, jd_text)
        self.index, self.chunks = build_index(structured_chunks)
        return len(self.chunks)

    def load(self):
        self.index, self.chunks = load_index()
        return self

    def query(self, query_text: str, top_k: int = 4,
              threshold: float = 1.8, source_filter: str = None):
        if self.index is None or not self.chunks:
            return []

        q_vec = embed_text(query_text).astype(np.float32)
        q_vec = np.expand_dims(q_vec, axis=0)

        k = min(top_k * 3, len(self.chunks))
        distances, indices = self.index.search(q_vec, k)

        results = []
        seen_texts = set()

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            if float(dist) > threshold:
                continue

            chunk = self.chunks[idx]

            if source_filter and chunk.get("source") != source_filter:
                continue

            text_key = chunk["text"][:80]
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)

            results.append({
                "label":   chunk["label"],
                "section": chunk["section"],
                "text":    chunk["text"],
                "source":  chunk["source"],
                "score":   round(float(dist), 4)
            })

            if len(results) >= top_k:
                break

        return results

    def compare_resume_to_jd(self, resume_text: str, jd_text: str) -> dict:
        resume_skills = set(extract_skills_from_text(resume_text))
        jd_skills     = set(extract_skills_from_text(jd_text))

        matched = resume_skills & jd_skills
        missing = jd_skills - resume_skills
        extra   = resume_skills - jd_skills

        total = len(jd_skills)
        score = round((len(matched) / total * 100) if total > 0 else 0, 1)

        jd_context = self.query(
            "required skills qualifications",
            top_k=3, source_filter="jd", threshold=2.0
        )
        resume_context = self.query(
            "skills experience technologies",
            top_k=3, source_filter="resume", threshold=2.0
        )

        return {
            "matched_skills":  sorted(matched),
            "missing_skills":  sorted(missing),
            "extra_skills":    sorted(extra),
            "match_score":     score,
            "jd_context":      jd_context,
            "resume_context":  resume_context,
            "total_jd_skills": total,
            "verdict": (
                "Strong match"    if score >= 70 else
                "Moderate match"  if score >= 40 else
                "Needs improvement"
            )
        }