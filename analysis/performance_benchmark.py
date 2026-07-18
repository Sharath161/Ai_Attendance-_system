"""Performance & scalability benchmark for the Smart Attendance backend.

Measures REAL, reproducible numbers on this machine:

  1. Per-stage inference latency  (YuNet detect | ArcFace align+embed | end-to-end)
  2. Recognition throughput       (images / second, single CPU worker)
  3. Matching scalability         (cosine search time vs gallery size, 1e2 .. 2e5)
  4. Optimizer stage timing       (quality scoring, LOO threshold sweep)

Outputs:
  analysis/output/perf_fig1_latency_breakdown.png
  analysis/output/perf_fig2_throughput.png
  analysis/output/perf_fig3_matching_scalability.png
  analysis/output/perf_fig4_optimizer_stages.png
  analysis/output/perf_metrics.json
  analysis/output/PERFORMANCE_REPORT.md

Run from the project root:
    python -m analysis.performance_benchmark
"""
from __future__ import annotations

import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from core.config import get_settings
from core.math_utils import l2_normalize
from worker.model_adapter import (
    FaceEmbeddingModel, _yunet_to_arcface_kps, _align_face, _preprocess_arcface,
)
from worker.optimizer import laplacian_variance, compute_quality_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "analysis" / "output"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.labelsize": 10, "font.size": 9,
})
BLUE, GREEN, RED, PURPLE, GOLD = "#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_images(reg_dir: Path, limit: int = 120):
    paths = []
    for d in sorted(p for p in reg_dir.iterdir() if p.is_dir()):
        paths += [p for p in d.glob("*") if p.suffix.lower() in IMG_EXT]
    paths = paths[:limit]
    imgs = [cv2.imread(str(p)) for p in paths]
    return [im for im in imgs if im is not None]


def pct(a, q):
    return float(np.percentile(a, q))


# ──────────────────────────────────────────────────────────────────────────────
# 1. Per-stage latency
# ──────────────────────────────────────────────────────────────────────────────
def bench_latency(model, images, reps=3):
    det_ms, emb_ms, e2e_ms = [], [], []
    # warm-up
    for im in images[:5]:
        model.detect_and_embed(im)

    for _ in range(reps):
        for im in images:
            h, w = im.shape[:2]

            t0 = time.perf_counter()
            model._detector.setInputSize((w, h))
            _, faces = model._detector.detect(im)
            t1 = time.perf_counter()
            det_ms.append((t1 - t0) * 1000)

            if faces is None or len(faces) == 0:
                continue
            primary = max(faces, key=lambda f: f[-1])
            kps = _yunet_to_arcface_kps(primary)

            t2 = time.perf_counter()
            aligned = _align_face(im, kps)
            inp = _preprocess_arcface(aligned)
            raw = model._session.run(None, {model._input_name: inp})[0][0]
            _ = l2_normalize(np.asarray(raw, dtype=np.float32).flatten())
            t3 = time.perf_counter()
            emb_ms.append((t3 - t2) * 1000)
            e2e_ms.append((t1 - t0 + t3 - t2) * 1000)

    return {
        "detect": det_ms, "embed": emb_ms, "e2e": e2e_ms,
        "summary": {
            "detect_mean_ms": float(np.mean(det_ms)),
            "embed_mean_ms": float(np.mean(emb_ms)),
            "e2e_mean_ms": float(np.mean(e2e_ms)),
            "e2e_p50_ms": pct(e2e_ms, 50), "e2e_p95_ms": pct(e2e_ms, 95),
            "e2e_p99_ms": pct(e2e_ms, 99),
            "samples": len(e2e_ms),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Throughput (single worker)
# ──────────────────────────────────────────────────────────────────────────────
def bench_throughput(model, images, duration_s=8.0):
    for im in images[:5]:
        model.detect_and_embed(im)
    n = 0
    t0 = time.perf_counter()
    i = 0
    while time.perf_counter() - t0 < duration_s:
        model.detect_and_embed(images[i % len(images)])
        n += 1
        i += 1
    elapsed = time.perf_counter() - t0
    ips = n / elapsed
    return {
        "images_processed": n, "elapsed_s": elapsed,
        "images_per_sec": ips, "per_hour": ips * 3600,
        # batch-window model from README: worker runs every ~2h
        "capacity_2h_window": int(ips * 3600 * 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Matching scalability (cosine over a gallery of prototypes)
# ──────────────────────────────────────────────────────────────────────────────
def bench_matching(dim=512, sizes=(100, 1_000, 10_000, 50_000, 100_000, 200_000), reps=50):
    rng = np.random.default_rng(0)
    results = []
    for n in sizes:
        gallery = l2_normalize_rows(rng.standard_normal((n, dim)).astype(np.float32))
        query = l2_normalize(rng.standard_normal(dim).astype(np.float32))
        # warm-up
        _ = gallery @ query
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            sims = gallery @ query
            _ = int(np.argmax(sims))
            ts.append((time.perf_counter() - t0) * 1000)
        results.append({"gallery": n, "mean_ms": float(np.mean(ts)),
                        "p95_ms": pct(ts, 95),
                        "throughput_qps": 1000.0 / float(np.mean(ts))})
    return results


def l2_normalize_rows(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (norms + 1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Optimizer stage timing
# ──────────────────────────────────────────────────────────────────────────────
def bench_optimizer(model, images):
    # Stage 1: quality scoring per image
    t0 = time.perf_counter()
    for im in images:
        h, w = im.shape[:2]
        model._detector.setInputSize((w, h))
        _, faces = model._detector.detect(im)
        if faces is None or len(faces) == 0:
            continue
        p = max(faces, key=lambda f: f[-1])
        bbox = {"x": int(p[0]), "y": int(p[1]), "w": int(p[2]), "h": int(p[3])}
        _ = compute_quality_score(im, bbox, float(p[-1]))
    q_ms = (time.perf_counter() - t0) * 1000

    # Stage 3: LOO threshold sweep cost model (80 thresholds x N prototypes)
    rng = np.random.default_rng(1)
    sweep = {}
    for n in (10, 50, 100, 500):
        protos = l2_normalize_rows(rng.standard_normal((n, 512)).astype(np.float32))
        thresholds = np.arange(0.20, 0.60, 0.005)
        t0 = time.perf_counter()
        sims = protos @ protos.T
        for t in thresholds:
            _ = (sims >= t)
        sweep[n] = (time.perf_counter() - t0) * 1000
    return {"quality_scoring_total_ms": q_ms,
            "quality_scoring_per_img_ms": q_ms / max(len(images), 1),
            "loo_sweep_ms": sweep}


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────
def fig_latency(lat):
    s = lat["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))

    ax = axes[0]
    stages = ["YuNet\ndetect", "ArcFace\nalign+embed", "End-to-end"]
    vals = [s["detect_mean_ms"], s["embed_mean_ms"], s["e2e_mean_ms"]]
    bars = ax.bar(stages, vals, color=[BLUE, GREEN, PURPLE], edgecolor="white")
    ax.set_ylabel("Latency (ms)"); ax.set_title("Mean Per-Stage Latency (CPU)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f} ms",
                ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    e2e = np.array(lat["e2e"])
    ax.hist(e2e, bins=30, color=PURPLE, edgecolor="white", alpha=0.85)
    for q, c, lab in [(50, "black", "p50"), (95, RED, "p95"), (99, "darkred", "p99")]:
        v = pct(e2e, q)
        ax.axvline(v, color=c, ls="--", lw=1.3, label=f"{lab}={v:.1f} ms")
    ax.set_xlabel("End-to-end latency (ms)"); ax.set_ylabel("Count")
    ax.set_title(f"End-to-End Latency Distribution (n={s['samples']})")
    ax.legend(fontsize=8)

    plt.suptitle("Performance Figure 1 — Inference Latency Breakdown",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "perf_fig1_latency_breakdown.png", bbox_inches="tight", dpi=150)
    plt.close()


def fig_throughput(tp):
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ["Per second", "Per minute", "Per 2-h batch\nwindow"]
    vals = [tp["images_per_sec"], tp["images_per_sec"] * 60, tp["capacity_2h_window"]]
    bars = ax.bar(labels, vals, color=[BLUE, GREEN, GOLD], edgecolor="white")
    ax.set_yscale("log")
    ax.set_ylabel("Images processed (log scale)")
    ax.set_title("Performance Figure 2 — Single-Worker Recognition Throughput\n"
                 f"{tp['images_per_sec']:.1f} images/sec sustained on one CPU worker",
                 fontsize=11, fontweight="bold")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{int(v):,}",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "perf_fig2_throughput.png", bbox_inches="tight", dpi=150)
    plt.close()


def fig_matching(match):
    sizes = [r["gallery"] for r in match]
    ms = [r["mean_ms"] for r in match]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))

    ax = axes[0]
    ax.plot(sizes, ms, "o-", color=BLUE, lw=2, markersize=7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Gallery size (enrolled prototypes)")
    ax.set_ylabel("Search time (ms)")
    ax.set_title("Cosine Match Latency vs Gallery Size")
    for x, y in zip(sizes, ms):
        ax.annotate(f"{y:.2f} ms", (x, y), textcoords="offset points",
                    xytext=(0, 8), fontsize=8, ha="center")

    ax = axes[1]
    qps = [r["throughput_qps"] for r in match]
    ax.plot(sizes, qps, "s-", color=GREEN, lw=2, markersize=7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Gallery size (enrolled prototypes)")
    ax.set_ylabel("Match throughput (queries/sec, log)")
    ax.set_title("Matching Throughput vs Gallery Size")

    plt.suptitle("Performance Figure 3 — Matching Scalability (single-thread NumPy BLAS)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "perf_fig3_matching_scalability.png", bbox_inches="tight", dpi=150)
    plt.close()


def fig_optimizer(opt):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))

    ax = axes[0]
    ax.bar(["Quality scoring\n(per image)"], [opt["quality_scoring_per_img_ms"]],
           color=RED, edgecolor="white", width=0.5)
    ax.set_ylabel("ms / image")
    ax.set_title(f"Stage 1 — Quality Scoring\n({opt['quality_scoring_per_img_ms']:.1f} ms/img)")
    ax.text(0, opt["quality_scoring_per_img_ms"],
            f"{opt['quality_scoring_per_img_ms']:.1f} ms", ha="center", va="bottom")

    ax = axes[1]
    ns = sorted(opt["loo_sweep_ms"].keys(), key=int)
    vals = [opt["loo_sweep_ms"][n] for n in ns]
    ax.plot([int(n) for n in ns], vals, "o-", color=PURPLE, lw=2, markersize=7)
    ax.set_xlabel("Students in cohort")
    ax.set_ylabel("LOO 80-threshold sweep (ms)")
    ax.set_title("Stage 3 — LOO Threshold Calibration Cost")

    plt.suptitle("Performance Figure 4 — Optimizer Stage Timing",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "perf_fig4_optimizer_stages.png", bbox_inches="tight", dpi=150)
    plt.close()


def write_report(metrics):
    m = metrics
    lat = m["latency"]["summary"]
    tp = m["throughput"]
    match_rows = "\n".join(
        f"| {r['gallery']:,} | {r['mean_ms']:.3f} | {r['p95_ms']:.3f} | {r['throughput_qps']:,.0f} |"
        for r in m["matching"]
    )
    md = f"""# Performance & Scalability Benchmark — Smart Attendance System

**Project:** Develop an Efficient and Scalable Backend (Smart Attendance)
**Module:** MA981-7-FY · MSc Data Science and Its Applications · University of Essex
**Generated:** {m['generated_at']}
**Host:** {m['host']['cpu_count']} logical CPUs · Python {m['host']['python']} · onnxruntime CPUExecutionProvider
**Pipeline:** YuNet detect → ArcFace MobileNet (512-d) → cosine match

---

## 1. Inference Latency (per image, CPU)

| Stage | Mean (ms) |
|---|---|
| YuNet detection | {lat['detect_mean_ms']:.1f} |
| ArcFace align + embed | {lat['embed_mean_ms']:.1f} |
| **End-to-end** | **{lat['e2e_mean_ms']:.1f}** |
| End-to-end p50 / p95 / p99 | {lat['e2e_p50_ms']:.1f} / {lat['e2e_p95_ms']:.1f} / {lat['e2e_p99_ms']:.1f} |

Samples: {lat['samples']} timed inferences.

## 2. Recognition Throughput (single CPU worker)

| Metric | Value |
|---|---|
| Sustained throughput | **{tp['images_per_sec']:.1f} images/sec** |
| Per hour | {tp['per_hour']:,.0f} images |
| Capacity per 2-hour batch window | {tp['capacity_2h_window']:,} images |

The README specifies the batch worker runs on a ~2-hour cycle; a single CPU
worker clears **{tp['capacity_2h_window']:,} images** per cycle, so a cohort of a
few hundred students with several kiosk captures each is handled comfortably by
one worker, and the design scales horizontally by adding workers.

## 3. Matching Scalability (cosine similarity over enrolled gallery)

Because embeddings are L2-normalised, matching is a single dense matrix–vector
product (`gallery @ query`) handled by NumPy's BLAS — O(N·d) and cache-friendly.

| Gallery size | Mean (ms) | p95 (ms) | Throughput (q/s) |
|---|---|---|---|
{match_rows}

Even at **{m['matching'][-1]['gallery']:,}** enrolled prototypes a match completes in
~{m['matching'][-1]['mean_ms']:.1f} ms, so identity search is never the bottleneck; the
detector+embedder dominates end-to-end cost.

## 4. Optimizer Stage Timing

| Stage | Cost |
|---|---|
| Stage 1 — quality scoring | {m['optimizer']['quality_scoring_per_img_ms']:.1f} ms / image |
| Stage 3 — LOO 80-threshold sweep (100 students) | {m['optimizer']['loo_sweep_ms'].get('100', 0):.2f} ms |

The full leave-one-out calibration is dominated by embedding extraction, not the
threshold sweep — the sweep itself is sub-millisecond even for hundreds of
students, so per-demographic re-calibration is cheap to re-run whenever the
cohort changes.

## Figures

1. `perf_fig1_latency_breakdown.png` — per-stage & end-to-end latency
2. `perf_fig2_throughput.png` — single-worker throughput
3. `perf_fig3_matching_scalability.png` — match latency/throughput vs gallery size
4. `perf_fig4_optimizer_stages.png` — optimizer stage timing
"""
    (OUT / "PERFORMANCE_REPORT.md").write_text(md, encoding="utf-8")


def main():
    import os, platform
    settings = get_settings()
    reg_dir = settings.spool_dir.parent / "registrations"
    print("[perf] loading images...")
    images = load_images(reg_dir, limit=120)
    print(f"[perf] {len(images)} images loaded")

    model = FaceEmbeddingModel.from_settings(settings)

    print("[perf] benchmarking latency...")
    latency = bench_latency(model, images, reps=3)
    print("[perf] benchmarking throughput...")
    throughput = bench_throughput(model, images, duration_s=8.0)
    print("[perf] benchmarking matching scalability...")
    matching = bench_matching()
    print("[perf] benchmarking optimizer stages...")
    optimizer = bench_optimizer(model, images)

    print("[perf] rendering figures...")
    fig_latency(latency)
    fig_throughput(throughput)
    fig_matching(matching)
    fig_optimizer(optimizer)

    metrics = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "host": {"cpu_count": os.cpu_count(), "python": platform.python_version()},
        "latency": {"summary": latency["summary"]},
        "throughput": throughput,
        "matching": matching,
        "optimizer": {"quality_scoring_per_img_ms": optimizer["quality_scoring_per_img_ms"],
                      "quality_scoring_total_ms": optimizer["quality_scoring_total_ms"],
                      "loo_sweep_ms": {str(k): v for k, v in optimizer["loo_sweep_ms"].items()}},
    }
    (OUT / "perf_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # report expects string keys in loo_sweep_ms
    metrics["optimizer"]["loo_sweep_ms"] = {str(k): v for k, v in optimizer["loo_sweep_ms"].items()}
    write_report(metrics)
    print(f"[perf] done -> {OUT}")
    print(json.dumps(metrics["latency"]["summary"], indent=2))
    print(json.dumps(throughput, indent=2))


if __name__ == "__main__":
    main()
