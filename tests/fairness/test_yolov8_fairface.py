"""Fairness unit tests for the YOLOv8 + ArcFace pipeline.

These tests assert that the face recognition pipeline meets minimum
per-group accuracy thresholds, catching regressions if the model or
threshold changes.

Test strategy
-------------
* Olivetti dataset (sklearn) — always available, no download needed.
  40 people × 10 grayscale images.  Enroll 7, test 3 augmented copies.
  We don't have racial ground truth for Olivetti, so we use it as an
  overall accuracy gate.

* FairFace — optional (5 GB download).
  Skipped automatically if ``work/fairface`` doesn't exist.
  Tests per-race F1 >= FAIRNESS_MIN_F1 and gap <= FAIRNESS_MAX_GAP.

Run:
    pytest tests/fairness/ -v
    pytest tests/fairness/ -v --fairface-dir work/fairface
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pytest
from sklearn.datasets import fetch_olivetti_faces

# ── constants ─────────────────────────────────────────────────────────────────

OVERALL_MIN_F1 = 0.80          # gate for the Olivetti overall test
FAIRNESS_MIN_F1 = 0.75         # every race group must meet this
FAIRNESS_MAX_GAP = 0.15        # no race can trail the best group by more than 15%

FAIRFACE_DIR = Path("work/fairface")


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    from core.config import get_settings
    from worker.model_adapter import FaceEmbeddingModel

    settings = get_settings()
    try:
        return FaceEmbeddingModel.from_settings(settings)
    except FileNotFoundError as exc:
        pytest.skip(f"Models not downloaded: {exc}")


@pytest.fixture(scope="module")
def settings():
    from core.config import get_settings
    return get_settings()


# ── helpers ───────────────────────────────────────────────────────────────────

def olivetti_to_canvas(face: np.ndarray, size: int = 200) -> np.ndarray:
    u8 = np.clip(face * 255, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(cv2.resize(u8, (size, size), interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_GRAY2BGR)
    canvas = np.full((480, 640, 3), 140, dtype=np.uint8)
    y0, x0 = (480 - size) // 2, (640 - size) // 2
    canvas[y0:y0+size, x0:x0+size] = bgr
    return canvas


def augment(image: np.ndarray, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    brightness = rng.uniform(0.85, 1.15)
    angle = rng.uniform(-8.0, 8.0)
    out = np.clip(image.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    h, w = out.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(out, M, (w, h), borderValue=(128, 128, 128))


# ── Olivetti tests ────────────────────────────────────────────────────────────

class TestOlivettiOverall:
    """Pipeline sanity-check using Olivetti faces."""

    @pytest.fixture(scope="class")
    def olivetti_results(self, model, settings):
        from core.math_utils import l2_normalize

        data = fetch_olivetti_faces(shuffle=False)
        ENROLL, TEST = 7, 3
        threshold = settings.match_threshold

        enrollments: dict[int, np.ndarray] = {}
        for pid in range(40):
            imgs = data.images[data.target == pid]
            embs = []
            for i in range(ENROLL):
                canvas = olivetti_to_canvas(imgs[i])
                emb, count, _ = model.detect_and_embed(canvas)
                if emb is not None and count >= 1:
                    embs.append(emb)
            if embs:
                avg = l2_normalize(np.mean(np.stack(embs), axis=0))
                enrollments[pid] = avg

        tp = fp = fn = 0
        for pid, enroll_emb in enrollments.items():
            imgs = data.images[data.target == pid]
            for i in range(ENROLL, ENROLL + TEST):
                aug = augment(olivetti_to_canvas(imgs[i]), seed=pid * 100 + i)
                q_emb, count, _ = model.detect_and_embed(aug)
                if count == 0 or q_emb is None:
                    fn += 1
                    continue
                best_pid, best_score = max(
                    ((p, float(np.dot(q_emb, e))) for p, e in enrollments.items()),
                    key=lambda x: x[1],
                )
                if best_score >= threshold and best_pid == pid:
                    tp += 1
                elif best_score >= threshold:
                    fp += 1
                else:
                    fn += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

    def test_overall_f1(self, olivetti_results):
        f1 = olivetti_results["f1"]
        assert f1 >= OVERALL_MIN_F1, (
            f"Pipeline F1 {f1:.3f} is below minimum {OVERALL_MIN_F1} on Olivetti faces. "
            "Check model paths and match_threshold."
        )

    def test_zero_cross_person_matches(self, olivetti_results):
        # False positives (matched but wrong person) should be near zero
        fp = olivetti_results["fp"]
        total = olivetti_results["tp"] + olivetti_results["fp"] + olivetti_results["fn"]
        fp_rate = fp / total if total else 0.0
        assert fp_rate <= 0.05, (
            f"Cross-person false-positive rate {fp_rate:.1%} exceeds 5%. "
            "Threshold may be too low."
        )


# ── FairFace tests ────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not FAIRFACE_DIR.exists(),
    reason=f"FairFace dataset not found at {FAIRFACE_DIR}. "
           "Download from https://github.com/joojs/fairface",
)
class TestFairFaceEquity:
    """Per-race fairness tests using the FairFace dataset."""

    SAMPLE_PER_RACE = 100

    @pytest.fixture(scope="class")
    def fairface_rec_results(self, model, settings):
        from tools.fairface_validation import load_fairface_split, measure_recognition_accuracy

        images_by_race = load_fairface_split(
            FAIRFACE_DIR, split="val", sample_per_race=self.SAMPLE_PER_RACE
        )
        return measure_recognition_accuracy(images_by_race, model, settings.match_threshold)

    @pytest.fixture(scope="class")
    def fairface_det_results(self, model):
        from tools.fairface_validation import load_fairface_split, measure_detection_rates

        images_by_race = load_fairface_split(
            FAIRFACE_DIR, split="val", sample_per_race=self.SAMPLE_PER_RACE
        )
        return measure_detection_rates(images_by_race, model)

    def test_per_race_detection_rate(self, fairface_det_results):
        below = [(r.race, r.detection_rate) for r in fairface_det_results if r.detection_rate < 0.80]
        assert not below, (
            "Detection rate below 80% for: "
            + ", ".join(f"{r} ({v:.1%})" for r, v in below)
        )

    def test_per_race_f1(self, fairface_rec_results):
        below = [
            (r.race, r.f1)
            for r in fairface_rec_results
            if r.sample_size >= 10 and r.f1 < FAIRNESS_MIN_F1
        ]
        assert not below, (
            f"F1 below {FAIRNESS_MIN_F1} for: "
            + ", ".join(f"{r} ({v:.3f})" for r, v in below)
        )

    def test_f1_fairness_gap(self, fairface_rec_results):
        eligible = [r for r in fairface_rec_results if r.sample_size >= 10]
        if len(eligible) < 2:
            pytest.skip("Not enough races with sufficient sample size")
        best = max(r.f1 for r in eligible)
        gaps = [(r.race, best - r.f1) for r in eligible if best - r.f1 > FAIRNESS_MAX_GAP]
        assert not gaps, (
            f"F1 gap > {FAIRNESS_MAX_GAP} (vs best={best:.3f}) for: "
            + ", ".join(f"{r} (gap={g:.3f})" for r, g in gaps)
            + "\n  Consider per-race thresholds in Settings.race_thresholds"
        )
