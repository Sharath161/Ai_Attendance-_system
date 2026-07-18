"""Fairness testing pipeline for the face recognition system.

Tests whether YOLOv8 + ArcFace performs equitably across racial groups.

Two test modes
--------------
olivetti (default, no extra data required)
    Uses sklearn's Olivetti faces (40 people × 10 images).  Demographic
    metadata is not available for Olivetti, so all 40 people are treated as
    one group; the test validates that the pipeline has consistent accuracy.

fairface (requires FairFace dataset)
    Download: https://github.com/joojs/fairface
    Measures detection rate and one-shot recognition accuracy per race.
    Also writes results to the ``fairness_metrics`` DB table.

Usage
-----
    # Quick olivetti sanity check
    python -m tools.fairness_test

    # Full fairface run (point at local dataset)
    python -m tools.fairness_test --dataset fairface --dataset-dir work/fairface

    # Use live API for recognition instead of direct model calls
    python -m tools.fairness_test --dataset fairface --dataset-dir work/fairface \\
        --api http://localhost:8000

    # Save results to DB (always on for fairface, optional for olivetti)
    python -m tools.fairness_test --save-results
"""
from __future__ import annotations

import argparse
import asyncio
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from sklearn.datasets import fetch_olivetti_faces


# ── Olivetti helpers ──────────────────────────────────────────────────────────

def olivetti_to_bgr(face: np.ndarray, size: int = 200) -> np.ndarray:
    """Convert 64×64 float Olivetti face → colour BGR image."""
    u8 = np.clip(face * 255, 0, 255).astype(np.uint8)
    big = cv2.resize(u8, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)


def augment(image: np.ndarray, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    brightness = rng.uniform(0.85, 1.15)
    angle = rng.uniform(-8.0, 8.0)
    out = np.clip(image.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    h, w = out.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(out, M, (w, h), borderValue=(128, 128, 128))


# ── report helpers ────────────────────────────────────────────────────────────

def print_table(headers: list[str], rows: list[list]) -> None:
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    sep = "  ".join("-" * w for w in widths)
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


def flag_bias(results, best_f1: float, warn_gap: float = 0.10) -> list[str]:
    warnings = []
    for r in results:
        if hasattr(r, "f1") and r.f1 < best_f1 - warn_gap:
            pct = f"{r.f1 * 100:.1f}%"
            best_pct = f"{best_f1 * 100:.1f}%"
            warnings.append(
                f"  [BIAS ALERT] {r.race}: F1={pct} is >10% below best ({best_pct})"
            )
    return warnings


# ── olivetti test ─────────────────────────────────────────────────────────────

def run_olivetti(model, threshold: float, save: bool) -> None:
    print("\n=== Olivetti Faces (sanity check) ===")
    data = fetch_olivetti_faces(shuffle=False)
    n_people = 40
    ENROLL, TEST = 7, 3

    enrollments: dict[int, np.ndarray] = {}
    tp = fp = fn = 0
    detected = 0

    t0 = time.perf_counter()
    for pid in range(n_people):
        imgs = data.images[data.target == pid]
        enroll_embs = []
        for i in range(ENROLL):
            bgr = olivetti_to_bgr(imgs[i])
            # Center on 640x480 canvas for better detection
            canvas = np.full((480, 640, 3), 140, dtype=np.uint8)
            y0, x0 = (480 - 200) // 2, (640 - 200) // 2
            canvas[y0:y0+200, x0:x0+200] = bgr
            emb, count, _ = model.detect_and_embed(canvas)
            if emb is not None and count >= 1:
                enroll_embs.append(emb)
        if enroll_embs:
            enrollments[pid] = np.mean(np.stack(enroll_embs), axis=0)
            # Re-normalise the averaged vector
            from core.math_utils import l2_normalize
            enrollments[pid] = l2_normalize(enrollments[pid])

    # Test phase
    for pid in range(n_people):
        if pid not in enrollments:
            fn += TEST
            continue
        imgs = data.images[data.target == pid]
        for i in range(ENROLL, ENROLL + TEST):
            bgr = olivetti_to_bgr(imgs[i])
            aug = augment(bgr, seed=pid * 100 + i)
            canvas = np.full((480, 640, 3), 140, dtype=np.uint8)
            y0, x0 = (480 - 200) // 2, (640 - 200) // 2
            canvas[y0:y0+200, x0:x0+200] = aug
            q_emb, count, _ = model.detect_and_embed(canvas)
            if count == 0 or q_emb is None:
                fn += 1
                continue
            detected += 1
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

    elapsed = time.perf_counter() - t0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    total_tests = n_people * TEST
    print(f"\nEnrolled: {len(enrollments)}/{n_people} people")
    print(f"Detection rate on test frames: {detected}/{total_tests} ({detected/total_tests*100:.1f}%)")
    print(f"Precision: {precision:.3f}   Recall: {recall:.3f}   F1: {f1:.3f}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Threshold used: {threshold}")


# ── fairface test ─────────────────────────────────────────────────────────────

def run_fairface(
    dataset_dir: Path,
    model,
    threshold: float,
    sample_per_race: int,
    save: bool,
) -> None:
    from tools.fairface_validation import (
        RACES,
        load_fairface_split,
        measure_detection_rates,
        measure_recognition_accuracy,
    )

    print("\n=== FairFace Fairness Test ===")
    print(f"Dataset: {dataset_dir}")
    print(f"Sample per race: {sample_per_race}  |  Threshold: {threshold}\n")

    images_by_race = load_fairface_split(
        dataset_dir, split="val", sample_per_race=sample_per_race
    )

    # ── detection rate ────────────────────────────────────────────────────────
    print("[1/2] Measuring detection rates ...")
    t0 = time.perf_counter()
    det_results = measure_detection_rates(images_by_race, model)
    print(f"  done in {time.perf_counter() - t0:.1f}s\n")

    print_table(
        ["Race", "Total", "Detected", "Det. Rate"],
        [
            [r.race, r.total, r.detected, f"{r.detection_rate * 100:.1f}%"]
            for r in det_results
        ],
    )

    # ── recognition accuracy ──────────────────────────────────────────────────
    print("\n[2/2] Measuring recognition accuracy (one-shot + augmentation) ...")
    t0 = time.perf_counter()
    rec_results = measure_recognition_accuracy(images_by_race, model, threshold)
    print(f"  done in {time.perf_counter() - t0:.1f}s\n")

    print_table(
        ["Race", "Enrolled", "TP", "FP", "FN", "Precision", "Recall", "F1"],
        [
            [
                r.race, r.sample_size, r.tp, r.fp, r.fn,
                f"{r.precision:.3f}", f"{r.recall:.3f}", f"{r.f1:.3f}",
            ]
            for r in rec_results
        ],
    )

    best_f1 = max((r.f1 for r in rec_results), default=0)
    warnings = flag_bias(rec_results, best_f1)
    if warnings:
        print("\n" + "\n".join(warnings))
        print("\n  Recommendation: consider per-race threshold tuning via Settings.race_thresholds")
    else:
        print("\n  No significant bias detected at 10% F1-gap threshold.")

    # ── save to DB ────────────────────────────────────────────────────────────
    if save:
        _save_fairness_metrics(rec_results, det_results, threshold, "fairface")


def _save_fairness_metrics(rec_results, det_results, threshold: float, dataset: str) -> None:
    import asyncio as _asyncio
    from core.database import SessionLocal
    from core.models import FairnessMetric

    det_by_race = {r.race: r.detection_rate for r in det_results}
    now = datetime.now(timezone.utc)

    async def _write():
        async with SessionLocal() as s:
            for r in rec_results:
                s.add(FairnessMetric(
                    test_date=now,
                    dataset=dataset,
                    race_group=r.race,
                    sample_size=r.sample_size,
                    detection_rate=det_by_race.get(r.race),
                    precision=r.precision if r.sample_size else None,
                    recall=r.recall if r.sample_size else None,
                    f1=r.f1 if r.sample_size else None,
                    threshold_used=threshold,
                ))
            await s.commit()
        print(f"\nFairness metrics saved to DB ({len(rec_results)} rows).")

    _asyncio.run(_write())


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test face recognition fairness across racial groups."
    )
    p.add_argument("--dataset", choices=["olivetti", "fairface"], default="olivetti")
    p.add_argument("--dataset-dir", type=Path, default=Path("work/fairface"),
                   help="Path to extracted FairFace dataset (required for --dataset fairface)")
    p.add_argument("--sample-per-race", type=int, default=200,
                   help="Number of images to sample per race (fairface only)")
    p.add_argument("--save-results", action="store_true",
                   help="Write results to the fairness_metrics table")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from core.config import get_settings
    from worker.model_adapter import FaceEmbeddingModel

    settings = get_settings()
    print(f"Loading model: {settings.worker_model_version}")
    model = FaceEmbeddingModel.from_settings(settings)
    threshold = settings.match_threshold

    if args.dataset == "olivetti":
        run_olivetti(model, threshold, args.save_results)
    else:
        run_fairface(args.dataset_dir, model, threshold, args.sample_per_race, args.save_results)

    print("\ndashboard → http://localhost:8501  (Fairness & Bias page)")


if __name__ == "__main__":
    main()
