"""Matching — multi-prototype 1:1 verify and 1:N identify over the gallery."""
from __future__ import annotations

import numpy as np


def score_subject(query: np.ndarray, matrix: np.ndarray) -> float:
    """Multi-prototype score = max cosine over the subject's embeddings."""
    return float(np.max(matrix @ query))


def identify(query: np.ndarray, gallery: dict[str, tuple], threshold: float, topk: int = 5) -> dict:
    if not gallery:
        return {"match": None, "candidates": [], "threshold": threshold}
    ranked = sorted(
        ({"subject_id": sid, "name": name, "score": score_subject(query, mat)}
         for sid, (name, mat) in gallery.items()),
        key=lambda c: -c["score"])
    best = ranked[0]
    match = best if best["score"] >= threshold else None
    return {"match": match, "candidates": ranked[:topk], "threshold": threshold}


def verify(query: np.ndarray, subject_matrix: np.ndarray, threshold: float) -> dict:
    score = score_subject(query, subject_matrix)
    return {"is_match": score >= threshold, "score": score, "threshold": threshold}
