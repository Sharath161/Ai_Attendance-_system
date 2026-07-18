"""FairFace dataset utilities for per-race fairness validation.

FairFace dataset: https://github.com/joojs/fairface
Download from: https://drive.google.com/file/d/1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86/view

Expected on-disk layout after extraction:
    <dataset_dir>/
        fairface_label_train.csv    (columns: file, age, gender, race, service_test)
        fairface_label_val.csv
        train/                      (images)
        val/                        (images)

FairFace race labels (as they appear in the CSV):
    White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, Latino_Hispanic

Usage:
    from tools.fairface_validation import load_fairface_split, RACES
    images_by_race = load_fairface_split(Path("work/fairface"), split="val")
"""
from __future__ import annotations

import io
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import requests

# Canonical race labels (normalised lowercase/underscored)
RACES = [
    "white",
    "black",
    "indian",
    "east_asian",
    "southeast_asian",
    "middle_eastern",
    "latino_hispanic",
]

# Maps FairFace CSV label → our canonical label
_LABEL_MAP: dict[str, str] = {
    "White": "white",
    "Black": "black",
    "Indian": "indian",
    "East Asian": "east_asian",
    "Southeast Asian": "southeast_asian",
    "Middle Eastern": "middle_eastern",
    "Latino_Hispanic": "latino_hispanic",
}


@dataclass
class FaceImage:
    path: Path
    race: str


def load_fairface_split(
    dataset_dir: Path,
    split: str = "val",
    sample_per_race: int = 200,
    seed: int = 42,
) -> dict[str, list[FaceImage]]:
    """Load ``sample_per_race`` images per race from a FairFace split.

    Args:
        dataset_dir: Root directory of the extracted FairFace dataset.
        split:       ``"train"`` or ``"val"``.
        sample_per_race: How many images to sample per race group.
        seed:        RNG seed for reproducibility.

    Returns:
        Dict mapping canonical race label → list of FaceImage.
    """
    import csv

    csv_path = dataset_dir / f"fairface_label_{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"FairFace label file not found: {csv_path}\n"
            "Download from https://github.com/joojs/fairface and extract to "
            f"{dataset_dir}"
        )

    by_race: dict[str, list[FaceImage]] = {r: [] for r in RACES}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            race_label = _LABEL_MAP.get(row["race"])
            if race_label is None:
                continue
            img_path = dataset_dir / row["file"]
            if img_path.exists():
                by_race[race_label].append(FaceImage(path=img_path, race=race_label))

    rng = random.Random(seed)
    return {
        race: rng.sample(imgs, min(sample_per_race, len(imgs)))
        for race, imgs in by_race.items()
    }


# ── detection-rate test ───────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    race: str
    total: int
    detected: int
    detection_rate: float


def measure_detection_rates(
    images_by_race: dict[str, list[FaceImage]],
    model,          # FaceEmbeddingModel
) -> list[DetectionResult]:
    """Run face detection on every sampled image and report the rate per race."""
    results = []
    for race, imgs in images_by_race.items():
        detected = 0
        for fi in imgs:
            image = cv2.imread(str(fi.path))
            if image is None:
                continue
            _, face_count, _ = model.detect_and_embed(image)
            if face_count >= 1:
                detected += 1
        total = len(imgs)
        rate = detected / total if total else 0.0
        results.append(DetectionResult(race=race, total=total, detected=detected, detection_rate=rate))
    return results


# ── recognition test (augmented one-shot) ────────────────────────────────────

@dataclass
class RecognitionResult:
    race: str
    sample_size: int
    tp: int = 0   # augmented version correctly matched back to enrollment image
    fp: int = 0   # matched but wrong person (shouldn't happen in one-shot)
    fn: int = 0   # augmented version not matched (below threshold or no_face)
    detection_rate: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


def _augment(image: np.ndarray, seed: int) -> np.ndarray:
    """Apply slight brightness + rotation for the recognition hold-out."""
    rng = random.Random(seed)
    brightness = rng.uniform(0.85, 1.15)
    angle = rng.uniform(-8.0, 8.0)
    out = np.clip(image.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    h, w = out.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(out, M, (w, h), borderValue=(128, 128, 128))


def measure_recognition_accuracy(
    images_by_race: dict[str, list[FaceImage]],
    model,
    threshold: float,
) -> list[RecognitionResult]:
    """One-shot recognition test: enroll one image, test augmented version.

    For each face image:
    1. Embed the original  → "enrollment" vector
    2. Embed an augmented copy → "query" vector
    3. Compare query against all enrollment vectors
    4. Count as TP if the closest match is the same person AND above threshold
    """
    from core.math_utils import l2_normalize

    results = []
    for race, imgs in images_by_race.items():
        enrollments: list[tuple[int, np.ndarray]] = []  # (idx, embedding)
        detected_original = 0
        detected_query = 0

        for i, fi in enumerate(imgs):
            image = cv2.imread(str(fi.path))
            if image is None:
                continue
            emb, count, _ = model.detect_and_embed(image)
            if count >= 1 and emb is not None:
                enrollments.append((i, emb))
                detected_original += 1

        tp = fp = fn = 0
        for i, enroll_emb in enrollments:
            image = cv2.imread(str(imgs[i].path))
            aug = _augment(image, seed=i * 1000)
            q_emb, q_count, _ = model.detect_and_embed(aug)
            if q_count == 0 or q_emb is None:
                fn += 1
                continue
            detected_query += 1

            # Find closest enrollment
            best_idx, best_score = -1, -1.0
            for j, e_emb in enrollments:
                score = float(np.dot(q_emb, e_emb))
                if score > best_score:
                    best_score = score
                    best_idx = j

            if best_score >= threshold and best_idx == i:
                tp += 1
            elif best_score >= threshold:
                fp += 1
            else:
                fn += 1

        total = len(enrollments)
        det_rate = detected_original / len(imgs) if imgs else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        results.append(
            RecognitionResult(
                race=race,
                sample_size=total,
                tp=tp, fp=fp, fn=fn,
                detection_rate=det_rate,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
    return results


# ── API-based recognition test (uses live server) ─────────────────────────────

def api_recognize(api_url: str, image: np.ndarray, camera_id: str = "fairness-test") -> dict:
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    files = {"image": ("test.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")}
    data = {"camera_id": camera_id}
    try:
        r = requests.post(f"{api_url}/checkin/recognize", data=data, files=files, timeout=15)
        return r.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def api_register(api_url: str, student_id: str, name: str, race: str, images: list[np.ndarray]) -> bool:
    files = []
    for idx, img in enumerate(images):
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        files.append(("images", (f"enroll_{idx}.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")))
    data = {
        "student_id": student_id,
        "name": name,
        "class_id": "fairness-test",
        "demographic_group": race,
    }
    try:
        r = requests.post(f"{api_url}/students/register", data=data, files=files, timeout=30)
        return r.status_code in (200, 202, 409)
    except Exception:
        return False
