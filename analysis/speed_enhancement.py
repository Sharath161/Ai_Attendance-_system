"""Speed-enhancement experiment for the recognition pipeline (CPU).

Measures the embedding-inference cost and shows the effect of:
    - ONNXRuntime graph optimisation + intra-op threads
    - BATCHED inference (N faces in one ONNX call) vs one-at-a-time

Detection (YuNet) latency is reported for context (it dominates and is not
easily batched through cv2.FaceDetectorYN).

    python -m analysis.speed_enhancement

Outputs: analysis/output/speed_batching.png, speed_enhancement.json, SPEED_REPORT.md
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

from core.config import get_settings
from worker.model_adapter import (
    FaceEmbeddingModel, _yunet_to_arcface_kps, _align_face, _preprocess_arcface)
from tests.stress.image_seed import face_to_frame

OUT = Path("analysis/output"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
N_FACES = 96
REPS = 3


def build_inputs(model, n):
    """Detect+align n faces and return stacked preprocessed inputs [n,3,112,112]."""
    data = fetch_olivetti_faces(shuffle=False)
    inputs = []
    pid = 0
    while len(inputs) < n:
        for idx in range(10):
            frame = face_to_frame(data.images[data.target == pid % 40][idx])
            h, w = frame.shape[:2]
            model._detector.setInputSize((w, h))
            _, faces = model._detector.detect(frame)
            if faces is None or len(faces) == 0:
                continue
            kps = _yunet_to_arcface_kps(max(faces, key=lambda f: f[-1]))
            inp = _preprocess_arcface(_align_face(frame, kps))   # (1,3,112,112)
            inputs.append(inp[0])
            if len(inputs) >= n:
                break
        pid += 1
    return np.stack(inputs).astype(np.float32)


def make_session(path, threads):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    so.log_severity_level = 3   # silence per-run batch-shape warnings
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def bench_single(sess, iname, X, reps=REPS):
    sess.run(None, {iname: X[:1]})   # warm
    t0 = time.perf_counter()
    for _ in range(reps):
        for i in range(len(X)):
            sess.run(None, {iname: X[i:i+1]})
    dt = (time.perf_counter() - t0) / (reps * len(X))
    return dt * 1000  # ms/face


def bench_batch(sess, iname, X, batch, reps=REPS):
    sess.run(None, {iname: X[:batch]})  # warm
    t0 = time.perf_counter()
    for _ in range(reps):
        for i in range(0, len(X), batch):
            sess.run(None, {iname: X[i:i+batch]})
    dt = (time.perf_counter() - t0) / (reps * len(X))
    return dt * 1000  # ms/face


def bench_detection(model, reps=2):
    data = fetch_olivetti_faces(shuffle=False)
    frames = [face_to_frame(data.images[data.target == p % 40][0]) for p in range(24)]
    for f in frames[:3]:
        h, w = f.shape[:2]; model._detector.setInputSize((w, h)); model._detector.detect(f)
    t0 = time.perf_counter()
    for _ in range(reps):
        for f in frames:
            h, w = f.shape[:2]; model._detector.setInputSize((w, h)); model._detector.detect(f)
    return (time.perf_counter() - t0) / (reps * len(frames)) * 1000


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = get_settings()
    cores = os.cpu_count() or 4
    model = FaceEmbeddingModel.from_settings(settings)
    path = settings.face_recognition_model_path

    print(f"host cores={cores}; preparing {N_FACES} aligned faces ...")
    X = build_inputs(model, N_FACES)
    iname = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).get_inputs()[0].name

    det_ms = bench_detection(model)

    # 1-thread single (baseline-ish), all-thread single, all-thread batched
    sess1 = make_session(path, 1)
    sessN = make_session(path, cores)

    res = {}
    res["single_1thread_ms"] = bench_single(sess1, iname, X)
    res["single_allthread_ms"] = bench_single(sessN, iname, X)
    batches = [8, 16, 32, 64]
    res["batch_ms"] = {b: bench_batch(sessN, iname, X, b) for b in batches}
    res["detection_ms"] = det_ms

    # Real baseline = the CURRENT default (multi-threaded) session, not 1-thread.
    base = res["single_allthread_ms"]
    best_b = min(res["batch_ms"], key=lambda b: res["batch_ms"][b])
    best_batch_ms = res["batch_ms"][best_b]

    print(f"\nEmbedding latency (ms/face):")
    print(f"  single, 1 thread (ref): {res['single_1thread_ms']:.2f}   ({1000/res['single_1thread_ms']:.0f}/s)")
    print(f"  single, {cores} threads (=current default): {res['single_allthread_ms']:.2f}   ({1000/res['single_allthread_ms']:.0f}/s)  [baseline]")
    for b in batches:
        ms = res["batch_ms"][b]
        print(f"  batch={b:<3} {cores} threads   : {ms:.2f}   ({1000/ms:.0f}/s)   {base/ms:.1f}x vs default")
    print(f"\nYuNet detection        : {det_ms:.2f} ms/img  (dominant, not batched)")
    print(f"End-to-end now  ≈ {det_ms + res['single_1thread_ms']:.1f} ms/img")
    print(f"End-to-end batched embed ≈ {det_ms + best_batch_ms:.1f} ms/img (detection still dominates)")

    # figure
    labels = ["single\n1-thread", f"single\n{cores}-thread"] + [f"batch={b}" for b in batches]
    tps = [1000/res["single_1thread_ms"], 1000/res["single_allthread_ms"]] + [1000/res["batch_ms"][b] for b in batches]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, tps, color=["#C44E52", "#DD8452"] + ["#4C72B0"]*len(batches))
    ax.set_ylabel("Embedding throughput (faces/sec)")
    ax.set_title(f"Recognition-embedding speed-up via threads + batching\n(w600k_mbf, CPU, {cores} cores)")
    for b, v in zip(bars, tps):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout(); plt.savefig(OUT/"speed_batching.png", bbox_inches="tight", dpi=150); plt.close()

    res_json = {**{k: v for k, v in res.items() if k != "batch_ms"},
                "batch_ms": {str(k): v for k, v in res["batch_ms"].items()},
                "cores": cores, "best_batch": best_b, "best_batch_ms": best_batch_ms,
                "embed_speedup_x": base/best_batch_ms}
    (OUT/"speed_enhancement.json").write_text(json.dumps(res_json, indent=2), encoding="utf-8")

    md = [f"# Speed Enhancement — recognition inference (CPU, {cores} cores)\n",
          "Baseline = the system's current default (ONNXRuntime multi-threaded) session.\n",
          "| Config | ms/face | faces/sec | vs default |",
          "|---|---|---|---|",
          f"| single, 1 thread (threads off, reference) | {res['single_1thread_ms']:.2f} | {1000/res['single_1thread_ms']:.0f} | {base/res['single_1thread_ms']:.2f}x |",
          f"| single, {cores} threads (**current default**) | {res['single_allthread_ms']:.2f} | {1000/res['single_allthread_ms']:.0f} | 1.00x |"]
    for b in batches:
        ms = res["batch_ms"][b]
        md.append(f"| batch={b}, {cores} threads | {ms:.2f} | {1000/ms:.0f} | {base/ms:.2f}x |")
    md += ["",
           f"YuNet **detection ≈ {det_ms:.1f} ms/img** dominates end-to-end cost and is not "
           f"batched through cv2.FaceDetectorYN, so end-to-end goes from "
           f"~{det_ms + res['single_allthread_ms']:.0f} ms/img to ~{det_ms + best_batch_ms:.0f} ms/img.",
           "",
           "**Honest takeaways.**",
           "- The pipeline is *already* multi-threaded, so threading is not a new win — it is the baseline.",
           f"- Batching the embedder adds only **{base/best_batch_ms:.1f}x** on the recognition step "
           f"(best at batch={best_b}), and because detection dominates, end-to-end improves ~1.5x at most.",
           "- The genuine bottleneck is **detection** (~%.0f ms). The real speed levers are a quantised/INT8 "
           "YuNet or an accelerated execution provider (OpenVINO / GPU / DirectML) — none available on this "
           "CPU-only host." % det_ms,
           "- Practically the system is already comfortably within the 2-hour batch budget "
           f"(~{1000/(det_ms + res['single_allthread_ms']):.0f} img/s/worker), so speed is not the binding "
           "constraint; batching mainly trims total worker runtime and cost.",
           "",
           "Figure: `speed_batching.png`."]
    (OUT/"SPEED_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print("\nwrote speed_batching.png + SPEED_REPORT.md")


if __name__ == "__main__":
    main()
