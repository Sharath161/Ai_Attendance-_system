"""Few-shot (low-data) enrolment tooling.

Two techniques that lift accuracy when only 1-5 enrolment photos exist:

1. Enrolment amplification — each aligned face crop is expanded into a small
   bank of photometric/geometric variants (flip, small rotations, brightness),
   each embedded separately. One photo becomes ~6 prototypes, giving the
   multi-prototype matcher pose/lighting coverage it would otherwise lack.

2. S-norm score normalisation — cosine scores are z-normalised against a fixed
   impostor cohort from both the query side and the subject side, making scores
   comparable across subjects (equalises per-class operating points, so one
   global threshold treats all enrolled classes evenly).
"""
from __future__ import annotations

import cv2
import numpy as np

from faceapi.engine import FaceEngine


# ── enrolment amplification ───────────────────────────────────────────────────
def _rot(crop: np.ndarray, deg: float) -> np.ndarray:
    M = cv2.getRotationMatrix2D((56, 56), deg, 1.0)
    return cv2.warpAffine(crop, M, (112, 112), borderValue=0)


def _bright(crop: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(crop.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def augment_bank(crop: np.ndarray) -> list[np.ndarray]:
    """Aligned 112x112 crop -> 6 variants (original included)."""
    return [
        crop,
        cv2.flip(crop, 1),
        _rot(crop, 8.0),
        _rot(crop, -8.0),
        _bright(crop, 1.15),
        _bright(crop, 0.85),
    ]


def amplified_embeddings(engine: FaceEngine, crop: np.ndarray, batch: int = 16) -> np.ndarray:
    """One aligned crop -> [n_aug, 512] unit embeddings."""
    return engine.embed_crops(augment_bank(crop), batch=batch)


# ── s-norm score normalisation ────────────────────────────────────────────────
class ScoreNormalizer:
    """Symmetric s-norm against a fixed impostor cohort.

    cohort : [M, 512] unit vectors of identities that are NOT enrolled.
    """

    def __init__(self, cohort: np.ndarray) -> None:
        self.cohort = np.asarray(cohort, dtype=np.float32)

    def _stats(self, v: np.ndarray) -> tuple[float, float]:
        s = self.cohort @ v
        return float(s.mean()), float(s.std() + 1e-6)

    def normalize(self, raw: float, query: np.ndarray, subject_proto: np.ndarray) -> float:
        mq, sq = self._stats(query)
        mp, sp = self._stats(subject_proto)
        return 0.5 * ((raw - mq) / sq + (raw - mp) / sp)
