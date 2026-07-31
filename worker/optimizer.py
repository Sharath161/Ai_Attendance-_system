"""Embedding optimizer — quality scoring, prototype aggregation, threshold calibration.

Three-stage hypertuning pipeline:

Stage 1 — Quality scoring
    Each registered image is scored on sharpness (Laplacian variance) and
    YuNet face-confidence.  Low-quality embeddings are down-weighted so a bad
    angle does not pollute the class prototype.

Stage 2 — Prototype aggregation
    All quality-weighted embeddings for a student are collapsed into a single
    512-d L2-normalised class prototype.  Matching against one prototype per
    student is faster and more accurate than nearest-neighbour over raw images.

Stage 3 — Threshold calibration
    Leave-one-out cross-validation over all registered students:
    • For every student, treat one embedding as the query and match against all
      other prototypes.
    • Sweep thresholds from 0.20 to 0.60 in 0.005 steps.
    • Find the threshold maximising macro-averaged F1 globally.
    • Repeat within each demographic group to find per-group thresholds.

Results are written back to the database:
    • StudentEmbedding.image_quality_score — per-image score
    • Prototype embeddings replace raw ones (active=False for old rows)
    • Settings.race_thresholds updated in .env (or returned for manual apply)

Run standalone:
    python -m worker.optimizer

Or call from API:
    POST /admin/optimize
"""
from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from sqlalchemy import select, update

from core.config import get_settings
from core.database import SessionLocal
from core.init_db import create_tables_for_local_dev
from core.models import Student, StudentEmbedding, StudentStatus
from worker.model_adapter import FaceEmbeddingModel


# ── Quality scoring ───────────────────────────────────────────────────────────

def laplacian_variance(image: np.ndarray) -> float:
    """Measure image sharpness: higher = sharper.  Blurry images score < 100."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_size_score(bbox: dict | None, image_shape: tuple) -> float:
    """Fraction of frame area covered by the face (0-1).  Larger is better."""
    if bbox is None:
        return 0.0
    h, w = image_shape[:2]
    face_area = bbox["w"] * bbox["h"]
    return min(face_area / (w * h), 1.0)


def compute_quality_score(
    image: np.ndarray,
    bbox: dict | None,
    yunet_confidence: float,
) -> float:
    """Combined quality score in [0, 1].

    Weights:
        40 % — image sharpness  (Laplacian variance, capped at 800)
        40 % — YuNet detection confidence
        20 % — face size (face area / frame area)
    """
    sharpness = min(laplacian_variance(image) / 800.0, 1.0)
    size      = face_size_score(bbox, image.shape)
    return 0.40 * sharpness + 0.40 * yunet_confidence + 0.20 * size


# ── Prototype aggregation ─────────────────────────────────────────────────────

def weighted_mean_prototype(
    embeddings: list[np.ndarray],
    weights: list[float],
) -> np.ndarray:
    """Compute a quality-weighted mean embedding, then L2-normalise."""
    from core.math_utils import l2_normalize
    w = np.array(weights, dtype=np.float32)
    w = w / w.sum()                          # normalise weights to sum 1
    stacked = np.stack(embeddings)           # (N, 512)
    proto = (stacked * w[:, np.newaxis]).sum(axis=0)
    return l2_normalize(proto)


# ── Threshold calibration (Leave-One-Out CV) ──────────────────────────────────

class ScoredPair(NamedTuple):
    query_student_id: str
    match_student_id: str | None   # None = no match above any threshold
    score: float
    demographic: str | None


def leave_one_out_scores(
    prototypes: dict[str, tuple[np.ndarray, str | None]],
    raw_embeddings: dict[str, list[tuple[np.ndarray, float, str | None]]],
) -> list[ScoredPair]:
    """For each raw embedding, match against all *other* prototypes.

    prototypes  : {student_id: (proto_embedding, demographic_group)}
    raw_embeddings: {student_id: [(embedding, quality, demographic), ...]}

    Returns a list of (query_student_id, best_match_student_id, best_score, demographic).
    """
    pairs: list[ScoredPair] = []
    student_ids = list(prototypes.keys())

    for true_sid, embs in raw_embeddings.items():
        for emb, _qual, demo in embs:
            best_sid: str | None = None
            best_score = -1.0
            for cand_sid, (proto, _) in prototypes.items():
                if cand_sid == true_sid:
                    continue       # leave-one-out: skip own prototype
                score = float(np.dot(emb, proto))
                if score > best_score:
                    best_score = score
                    best_sid = cand_sid
            pairs.append(ScoredPair(true_sid, best_sid, best_score, demo))

    return pairs


def sweep_threshold(
    pairs: list[ScoredPair],
    student_ids: list[str],
    lo: float = 0.20,
    hi: float = 0.60,
    step: float = 0.005,
) -> tuple[float, dict]:
    """Sweep thresholds and return (best_threshold, metrics_at_best).

    For each threshold t:
        - A query is TP if  score >= t  AND  best_match == true_student
        - A query is FP if  score >= t  AND  best_match != true_student
        - A query is FN if  score <  t  (i.e., rejected)
    Macro-averaged F1 across all students is the optimisation target.
    """
    sid_set = set(student_ids)
    thresholds = np.arange(lo, hi + step / 2, step)

    best_t = lo
    best_f1 = -1.0
    best_metrics: dict = {}

    for t in thresholds:
        per_student: dict[str, dict[str, int]] = {
            sid: {"tp": 0, "fp": 0, "fn": 0} for sid in sid_set
        }
        for p in pairs:
            if p.score >= t and p.match_student_id == p.query_student_id:
                per_student[p.query_student_id]["tp"] += 1
            elif p.score >= t and p.match_student_id != p.query_student_id:
                # mismatch — counts as both FP for matched and FN for true
                per_student[p.query_student_id]["fn"] += 1
                if p.match_student_id and p.match_student_id in per_student:
                    per_student[p.match_student_id]["fp"] += 1
            else:
                per_student[p.query_student_id]["fn"] += 1

        f1s = []
        for s in per_student.values():
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)

        macro_f1 = float(np.mean(f1s)) if f1s else 0.0
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_t  = float(t)
            best_metrics = {
                "threshold": best_t,
                "macro_f1": round(best_f1, 4),
                "n_students": len(f1s),
                "per_student_f1": [round(v, 4) for v in f1s],
            }

    return best_t, best_metrics


def calibrate_per_demographic(
    pairs: list[ScoredPair],
    student_ids: list[str],
) -> dict[str, float]:
    """Find best threshold per demographic group independently."""
    group_pairs: dict[str, list[ScoredPair]] = defaultdict(list)
    group_sids:  dict[str, set[str]]         = defaultdict(set)

    for p in pairs:
        if p.demographic:
            group_pairs[p.demographic].append(p)
            group_sids[p.demographic].add(p.query_student_id)

    race_thresholds: dict[str, float] = {}
    for group, gp in group_pairs.items():
        sids = list(group_sids[group])
        if len(sids) < 2:
            continue    # need at least 2 students to do LOO
        t, metrics = sweep_threshold(gp, sids)
        race_thresholds[group] = round(t, 4)
        print(f"[opt] demographic={group} best_t={t:.3f} f1={metrics.get('macro_f1', '?')}")

    return race_thresholds


# ── Main pipeline ─────────────────────────────────────────────────────────────

class OptimizationResult(NamedTuple):
    global_threshold: float
    race_thresholds: dict[str, float]
    global_f1: float
    students_processed: int
    embeddings_scored: int
    prototypes_built: int
    per_demographic_f1: dict[str, float]
    low_quality_count: int


async def run_optimization(min_quality: float = 0.15) -> OptimizationResult:
    """Full three-stage optimization pipeline.

    Args:
        min_quality: embeddings below this quality score are excluded from the
                     prototype (but kept in DB for audit).
    """
    await create_tables_for_local_dev()
    settings = get_settings()
    model = FaceEmbeddingModel.from_settings(settings)

    reg_base = settings.spool_dir.parent / "registrations"

    async with SessionLocal() as session:
        # Load all active students
        result = await session.execute(
            select(Student).where(Student.status == StudentStatus.active)
        )
        students = result.scalars().all()

        if not students:
            print("[opt] No active students found — register students first.")
            return OptimizationResult(
                global_threshold=settings.match_threshold,
                race_thresholds={},
                global_f1=0.0,
                students_processed=0,
                embeddings_scored=0,
                prototypes_built=0,
                per_demographic_f1={},
                low_quality_count=0,
            )

        print(f"[opt] Processing {len(students)} active students")

        # ── Stage 1: quality-score every registration image ───────────────────
        student_data: dict[str, dict] = {}     # sid -> {proto, raw_embs, demo}
        total_scored = 0
        total_low_q  = 0

        for student in students:
            reg_dir = reg_base / student.student_id
            if not reg_dir.exists():
                continue

            images = sorted(
                p for p in reg_dir.glob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )
            if not images:
                continue

            demo_group: str | None = None
            meta = reg_dir / "metadata.json"
            if meta.exists():
                try:
                    demo_group = json.loads(meta.read_text()).\
                        get("demographic_group")
                except Exception:
                    pass

            scored: list[tuple[np.ndarray, float]] = []

            for img_path in images:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Run detection to get confidence + bbox
                h, w = img.shape[:2]
                model._detector.setInputSize((w, h))
                _, faces = model._detector.detect(img)

                if faces is None or len(faces) == 0:
                    continue

                primary    = max(faces, key=lambda f: f[-1])
                confidence = float(primary[-1])
                bbox       = {"x": int(primary[0]), "y": int(primary[1]),
                              "w": int(primary[2]), "h": int(primary[3])}

                quality = compute_quality_score(img, bbox, confidence)
                total_scored += 1

                if quality < min_quality:
                    total_low_q += 1
                    print(f"[opt]   skip low-quality image: {img_path.name} q={quality:.3f}")
                    continue

                # Get embedding (re-use detect_and_embed which does alignment)
                emb, _, _ = model.detect_and_embed(img)
                if emb is None:
                    continue

                # Update quality score in DB
                await session.execute(
                    update(StudentEmbedding)
                    .where(
                        StudentEmbedding.student_id == student.id,
                        StudentEmbedding.active.is_(True),
                    )
                    .values(image_quality_score=quality)
                )
                scored.append((emb, quality))

            if not scored:
                continue

            student_data[student.student_id] = {
                "student_id": student.student_id,
                "db_id":      student.id,
                "demo":       demo_group,
                "scored":     scored,   # [(embedding, quality), ...]
            }

        # ── Stage 2: compute quality-weighted prototypes ──────────────────────
        prototypes: dict[str, tuple[np.ndarray, str | None]] = {}
        raw_for_loo: dict[str, list[tuple[np.ndarray, float, str | None]]] = {}
        built = 0

        for sid, data in student_data.items():
            embs    = [e for e, _ in data["scored"]]
            weights = [q for _, q in data["scored"]]
            demo    = data["demo"]

            proto = weighted_mean_prototype(embs, weights)
            prototypes[sid] = (proto, demo)
            raw_for_loo[sid] = [(e, q, demo) for e, q in data["scored"]]

            # Deactivate old raw embeddings and insert one prototype embedding
            await session.execute(
                update(StudentEmbedding)
                .where(
                    StudentEmbedding.student_id == data["db_id"],
                    StudentEmbedding.active.is_(True),
                )
                .values(active=False)
            )
            session.add(StudentEmbedding(
                student_id=data["db_id"],
                embedding=proto.tolist(),
                model_version=model.model_version + "-prototype",
                image_quality_score=float(np.mean(weights)),
                active=True,
                demographic_group=demo,
            ))
            built += 1

        await session.commit()

        if len(prototypes) < 2:
            print("[opt] Need ≥ 2 students for threshold calibration.")
            return OptimizationResult(
                global_threshold=settings.match_threshold,
                race_thresholds={},
                global_f1=0.0,
                students_processed=len(student_data),
                embeddings_scored=total_scored,
                prototypes_built=built,
                per_demographic_f1={},
                low_quality_count=total_low_q,
            )

        # ── Stage 3: leave-one-out threshold calibration ──────────────────────
        print(f"[opt] Running leave-one-out CV over {len(prototypes)} students...")
        pairs = leave_one_out_scores(prototypes, raw_for_loo)

        best_t, global_metrics = sweep_threshold(pairs, list(prototypes.keys()))
        print(f"[opt] Global best threshold: {best_t:.3f}  F1={global_metrics['macro_f1']}")

        race_thresholds = calibrate_per_demographic(pairs, list(prototypes.keys()))

        # ── Per-demographic F1 at best global threshold ───────────────────────
        demo_f1: dict[str, float] = {}
        demo_pairs: dict[str, list[ScoredPair]] = defaultdict(list)
        demo_sids:  dict[str, set[str]]         = defaultdict(set)
        for p in pairs:
            if p.demographic:
                demo_pairs[p.demographic].append(p)
                demo_sids[p.demographic].add(p.query_student_id)

        for group, gp in demo_pairs.items():
            t = race_thresholds.get(group, best_t)
            tp = sum(1 for p in gp if p.score >= t and p.match_student_id == p.query_student_id)
            fn = sum(1 for p in gp if p.score <  t)
            fp = sum(1 for p in gp if p.score >= t and p.match_student_id != p.query_student_id)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            demo_f1[group] = round(f1, 4)

        return OptimizationResult(
            global_threshold=round(best_t, 4),
            race_thresholds=race_thresholds,
            global_f1=global_metrics["macro_f1"],
            students_processed=len(student_data),
            embeddings_scored=total_scored,
            prototypes_built=built,
            per_demographic_f1=demo_f1,
            low_quality_count=total_low_q,
        )


def _write_thresholds_to_env(
    global_threshold: float,
    race_thresholds: dict[str, float],
    env_path: Path = Path(".env"),
) -> None:
    """Patch .env in-place with the calibrated thresholds."""
    if not env_path.exists():
        return
    lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines = []
    wrote_global = False
    wrote_race   = False
    for line in lines:
        if line.startswith("MATCH_THRESHOLD="):
            new_lines.append(f"MATCH_THRESHOLD={global_threshold}\n")
            wrote_global = True
        elif line.startswith("RACE_THRESHOLDS=") or \
             line.startswith("# RACE_THRESHOLDS="):
            new_lines.append(
                f"RACE_THRESHOLDS={json.dumps(race_thresholds)}\n"
            )
            wrote_race = True
        else:
            new_lines.append(line)
    if not wrote_global:
        new_lines.append(f"MATCH_THRESHOLD={global_threshold}\n")
    if not wrote_race and race_thresholds:
        new_lines.append(f"RACE_THRESHOLDS={json.dumps(race_thresholds)}\n")
    env_path.write_text("".join(new_lines), encoding="utf-8")
    print(f"[opt] Updated {env_path} with calibrated thresholds")


if __name__ == "__main__":
    import sys
    min_q = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15

    result = asyncio.run(run_optimization(min_quality=min_q))

    print()
    print("=" * 60)
    print("  Optimization Results")
    print("=" * 60)
    print(f"  Students processed  : {result.students_processed}")
    print(f"  Images scored       : {result.embeddings_scored}")
    print(f"  Low-quality dropped : {result.low_quality_count}")
    print(f"  Prototypes built    : {result.prototypes_built}")
    print(f"  Global threshold    : {result.global_threshold}")
    print(f"  Global F1           : {result.global_f1}")
    if result.race_thresholds:
        print(f"  Per-demographic thresholds:")
        for g, t in sorted(result.race_thresholds.items()):
            f1 = result.per_demographic_f1.get(g, "?")
            print(f"    {g:<22} t={t:.3f}  F1={f1}")
    print("=" * 60)

    # Patch .env so the API picks up calibrated thresholds on next restart
    root = Path(__file__).parent.parent
    _write_thresholds_to_env(
        result.global_threshold,
        result.race_thresholds,
        env_path=root / ".env",
    )
    print()
    print("Thresholds written to .env — restart the API to apply.")
