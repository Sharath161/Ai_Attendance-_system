"""Produce an INT8-quantized recognition model for low-resource deployment,
and verify accuracy parity + measure size/speed.

    python -m faceapi.quantize

Writes work/models/w600k_mbf.int8.onnx and prints a size/latency/parity report.
Dynamic quantization needs no calibration data and is CPU-friendly.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from faceapi.config import get_settings
from faceapi.engine import FaceEngine, _preprocess, _align, _yunet_to_arcface_kps


def _sample_inputs(engine: FaceEngine, n: int = 40) -> np.ndarray:
    from sklearn.datasets import fetch_olivetti_faces
    from faceapi._testutil import face_to_frame
    data = fetch_olivetti_faces(shuffle=False)
    xs = []
    pid = 0
    while len(xs) < n:
        for idx in range(10):
            frame = face_to_frame(data.images[data.target == pid % 40][idx])
            h, w = frame.shape[:2]
            engine._detector.setInputSize((w, h))
            _, faces = engine._detector.detect(frame)
            if faces is None or len(faces) == 0:
                continue
            kps = _yunet_to_arcface_kps(max(faces, key=lambda f: f[-1]))
            xs.append(_preprocess(_align(frame, kps))[0])
            if len(xs) >= n:
                break
        pid += 1
    return np.stack(xs).astype(np.float32)


def _bench(path: Path, X: np.ndarray, reps: int = 3) -> float:
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    sess.run(None, {iname: X[:1]})
    t0 = time.perf_counter()
    for _ in range(reps):
        for i in range(len(X)):
            sess.run(None, {iname: X[i:i+1]})
    return (time.perf_counter() - t0) / (reps * len(X)) * 1000


def _embeddings(path: Path, X: np.ndarray) -> np.ndarray:
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    out = []
    for i in range(len(X)):
        v = sess.run(None, {iname: X[i:i+1]})[0][0]
        out.append(v / (np.linalg.norm(v) + 1e-9))
    return np.stack(out)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    s = get_settings()
    full = Path(s.recognition_model_full)
    int8 = Path(s.recognition_model_int8)
    if not full.exists():
        print(f"missing full model: {full}"); sys.exit(1)

    print(f"quantizing {full.name} -> {int8.name} (dynamic INT8) ...")
    quantize_dynamic(str(full), str(int8), weight_type=QuantType.QInt8)

    size_full = full.stat().st_size / 1e6
    size_int8 = int8.stat().st_size / 1e6

    engine = FaceEngine(s.detection_model_path, full)
    X = _sample_inputs(engine, 40)

    ms_full = _bench(full, X)
    ms_int8 = _bench(int8, X)

    ef = _embeddings(full, X)
    eq = _embeddings(int8, X)
    cos = np.sum(ef * eq, axis=1)   # per-face cosine between full and int8 embedding

    print("\n" + "=" * 56)
    print("  INT8 QUANTIZATION REPORT (recognition model)")
    print("=" * 56)
    print(f"  size      : {size_full:.1f} MB  ->  {size_int8:.1f} MB   ({size_full/size_int8:.1f}x smaller)")
    print(f"  latency   : {ms_full:.2f} ms  ->  {ms_int8:.2f} ms/face   ({ms_full/ms_int8:.2f}x)")
    print(f"  embedding parity vs full model (cosine): "
          f"min={cos.min():.4f} mean={cos.mean():.4f}")
    print("=" * 56)
    print("  Enable it with FACEAPI_LOW_RESOURCE=true (uses the INT8 model).")


if __name__ == "__main__":
    main()
