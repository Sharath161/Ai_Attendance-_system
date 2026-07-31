import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = l2_normalize(left.astype(np.float32))
    right_norm = l2_normalize(right.astype(np.float32))
    return float(np.dot(left_norm, right_norm))


def best_cosine_match(
    query_embedding: np.ndarray,
    candidate_embeddings: list[tuple[str, np.ndarray]],
) -> tuple[str | None, float]:
    best_id: str | None = None
    best_score = -1.0

    for candidate_id, embedding in candidate_embeddings:
        score = cosine_similarity(query_embedding, embedding)
        if score > best_score:
            best_id = candidate_id
            best_score = score

    return best_id, best_score
