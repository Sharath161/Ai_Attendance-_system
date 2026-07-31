"""Helpers for tests, the smoke demo and benchmarks.

Turns small grayscale/colour face arrays (e.g. sklearn's Olivetti or LFW
samples) into realistic camera-style frames the detector can work with, plus
deterministic per-image augmentation so runs are reproducible.

Not imported by the service itself — test/benchmark support only.
"""
from __future__ import annotations

import random

import cv2
import numpy as np


def face_to_frame(
    face: np.ndarray,
    face_size: int = 280,
    canvas_w: int = 640,
    canvas_h: int = 480,
    brightness: float = 1.0,
    rotate_deg: float = 0.0,
) -> np.ndarray:
    """Small face array -> 640x480 uint8 BGR frame with the face centred.

    Accepts float [0,1] or uint8 input, grayscale (H,W) or colour (H,W,3).
    """
    arr = np.asarray(face)
    if arr.dtype != np.uint8:
        scale = 255.0 if arr.max() <= 1.0 + 1e-6 else 1.0
        arr = np.clip(arr * scale * brightness, 0, 255).astype(np.uint8)
    elif brightness != 1.0:
        arr = np.clip(arr.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

    big = cv2.resize(arr, (face_size, face_size), interpolation=cv2.INTER_LANCZOS4)
    face_bgr = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR) if big.ndim == 2 else big

    if rotate_deg:
        c = face_size // 2
        M = cv2.getRotationMatrix2D((c, c), rotate_deg, 1.0)
        face_bgr = cv2.warpAffine(face_bgr, M, (face_size, face_size),
                                  borderValue=(140, 140, 140))

    canvas = np.full((canvas_h, canvas_w, 3), 140, dtype=np.uint8)
    y0 = (canvas_h - face_size) // 2
    x0 = (canvas_w - face_size) // 2
    canvas[y0:y0 + face_size, x0:x0 + face_size] = face_bgr
    return canvas


def to_jpeg_bytes(frame: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def augment_params(seed: int) -> tuple[float, float]:
    """Deterministic (brightness, rotation_degrees) for a given seed."""
    rng = random.Random(seed)
    return rng.uniform(0.85, 1.15), rng.uniform(-6.0, 6.0)
