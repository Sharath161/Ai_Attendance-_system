"""Accuracy-enhancement experiment for the recognition model.

Evaluates recognition on a controlled 40-identity protocol (Olivetti:
enrol 7 images / test 3 per person) and compares enhancement strategies:

    S0  baseline      — mean prototype, single embedding, global t=0.35
    S1  +flip TTA     — average embedding of face + its mirror
    S2  +quality-wt   — quality-weighted prototype (down-weight blurry shots)
    S3  multi-proto   — keep all enrol embeddings, score = max similarity
    +   recalibrated threshold via DET curve at a target false-match-rate

Metrics: top-1 identification accuracy, EER, TAR@FMR=1%, best-F1 (+threshold).
Optionally compares two backbones (mbf vs r50) if both ONNX files exist.

    python -m analysis.accuracy_enhancement

Outputs to analysis/output/: acc_det_curve.png, acc_scores_hist.png,
acc_strategy_bars.png, acc_metrics.json, ACCURACY_REPORT.md
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

warnings.filterwarnings("ignore")

from core.config import get_settings
from core.math_utils import l2_normalize
from worker.model_adapter import FaceEmbeddingModel
from worker.optimizer import compute_quality_score
from tests.stress.image_seed import face_to_frame, augment_params

OUT = Path("analysis/output"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})

N_PEOPLE = 40
ENROLL = 7
TEST = 3
GLOBAL_T = 0.35
FMR_TARGET = 0.01


# ── embedding ────────────────────────────────────────────────────────────────
def embed(model, frame, tta=False):
    e, fc, bbox = model.detect_and_embed(frame)
    if e is None:
        return None, 0.0, None
    if tta:
        ef, fcf, _ = model.detect_and_embed(cv2.flip(frame, 1))
        if ef is not None:
            e = l2_normalize((e + ef) / 2.0)
    # quality (needs YuNet confidence)
    h, w = frame.shape[:2]
    model._detector.setInputSize((w, h))
    _, faces = model._detector.detect(frame)
    conf = float(max(faces, key=lambda f: f[-1])[-1]) if faces is not None and len(faces) else 0.0
    q = compute_quality_score(frame, bbox, conf) if bbox else 0.0
    return e.astype(np.float32), q, bbox


def build_cache(model, data, tta):
    """Return per-person dict of enrol/test embeddings (+quality)."""
    cache = {}
    for pid in range(N_PEOPLE):
        imgs = data.images[data.target == pid]
        enrol, test = [], []
        for idx in range(ENROLL):
            b, a = augment_params(pid * 100 + idx)
            e, q, _ = embed(model, face_to_frame(imgs[idx], brightness=b, rotate_deg=a), tta)
            if e is not None:
                enrol.append((e, q))
        for idx in range(ENROLL, ENROLL + TEST):
            b, a = augment_params(pid * 100 + idx)
            e, _, _ = embed(model, face_to_frame(imgs[idx], brightness=b, rotate_deg=a), tta)
            if e is not None:
                test.append(e)
        cache[pid] = {"enrol": enrol, "test": test}
    return cache


# ── gallery strategies ───────────────────────────────────────────────────────
def mean_prototype(enrol):
    return l2_normalize(np.mean([e for e, _ in enrol], axis=0))

def qw_prototype(enrol):
    embs = np.stack([e for e, _ in enrol]); w = np.array([max(q, 1e-3) for _, q in enrol])
    w = w / w.sum()
    return l2_normalize((embs * w[:, None]).sum(0))


def score_person(query, rep, multi):
    if multi:
        return max(float(np.dot(query, e)) for e in rep)   # rep = list of embeddings
    return float(np.dot(query, rep))                        # rep = one prototype


def evaluate_strategy(cache, proto_fn, multi=False):
    """Return genuine scores, impostor scores, top-1 accuracy."""
    gallery = {pid: ([e for e, _ in c["enrol"]] if multi else proto_fn(c["enrol"]))
               for pid, c in cache.items()}
    genuine, impostor = [], []
    top1 = total = 0
    for pid, c in cache.items():
        for q in c["test"]:
            scores = {p: score_person(q, rep, multi) for p, rep in gallery.items()}
            genuine.append(scores[pid])
            impostor.extend(s for p, s in scores.items() if p != pid)
            pred = max(scores, key=scores.get)
            top1 += (pred == pid); total += 1
    return np.array(genuine), np.array(impostor), top1 / total


# ── metrics ──────────────────────────────────────────────────────────────────
def det_metrics(gen, imp):
    ts = np.linspace(0.0, 1.0, 400)
    fmr = np.array([(imp >= t).mean() for t in ts])      # false match rate
    fnmr = np.array([(gen < t).mean() for t in ts])      # false non-match rate
    eer_i = np.argmin(np.abs(fmr - fnmr)); eer = (fmr[eer_i] + fnmr[eer_i]) / 2
    # TAR at target FMR
    ok = np.where(fmr <= FMR_TARGET)[0]
    if len(ok):
        t_at = ts[ok[0]]; tar = 1 - (gen < t_at).mean()
    else:
        t_at, tar = 1.0, 0.0
    # best-F1 threshold
    bestF1, bestT = 0.0, GLOBAL_T
    for t in ts:
        tp = (gen >= t).sum(); fp = (imp >= t).sum(); fn = (gen < t).sum()
        p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        if f1 > bestF1: bestF1, bestT = f1, t
    # F1 at global threshold
    tp = (gen >= GLOBAL_T).sum(); fp = (imp >= GLOBAL_T).sum(); fn = (gen < GLOBAL_T).sum()
    p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
    f1g = 2 * p * r / (p + r) if p + r else 0
    return {"eer": float(eer), "tar_at_fmr1pct": float(tar), "t_at_fmr1pct": float(t_at),
            "best_f1": float(bestF1), "best_t": float(bestT), "f1_global": float(f1g),
            "fmr": fmr, "fnmr": fnmr, "ts": ts}


# ── run all strategies for one model ─────────────────────────────────────────
def run_model(model, label):
    print(f"\n=== {label} ===")
    print("  caching embeddings (single + TTA) ...")
    t0 = time.perf_counter()
    cache_plain = build_cache(model, DATA, tta=False)
    cache_tta = build_cache(model, DATA, tta=True)
    embt = (time.perf_counter() - t0)

    strategies = {
        "S0 baseline (mean proto)":       (cache_plain, mean_prototype, False),
        "S1 +flip TTA":                   (cache_tta,   mean_prototype, False),
        "S2 +quality-weighted proto":     (cache_tta,   qw_prototype,   False),
        "S3 multi-prototype (max-sim)":   (cache_tta,   None,           True),
    }
    results = {}
    for name, (cache, fn, multi) in strategies.items():
        gen, imp, top1 = evaluate_strategy(cache, fn, multi)
        m = det_metrics(gen, imp)
        m.update({"top1": float(top1), "gen": gen, "imp": imp})
        results[name] = m
        print(f"  {name:<32} top1={top1*100:5.1f}%  EER={m['eer']*100:4.1f}%  "
              f"TAR@FMR1%={m['tar_at_fmr1pct']*100:5.1f}%  bestF1={m['best_f1']:.3f}@t={m['best_t']:.2f}")
    return results, embt


# ── figures ──────────────────────────────────────────────────────────────────
def fig_det(all_results, model_label):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, m in all_results.items():
        ax.plot(m["fmr"] * 100, m["fnmr"] * 100, lw=2, label=f"{name} (EER {m['eer']*100:.1f}%)")
    ax.plot([0, 100], [0, 100], "k:", lw=0.8)
    ax.set_xscale("log"); ax.set_xlim(0.05, 100); ax.set_ylim(0, 60)
    ax.set_xlabel("False Match Rate (%, log)"); ax.set_ylabel("False Non-Match Rate (%)")
    ax.set_title(f"DET Curve — recognition strategies ({model_label})\nlower-left is better")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout(); plt.savefig(OUT / "acc_det_curve.png", bbox_inches="tight", dpi=150); plt.close()


def fig_hist(base, best, base_name, best_name):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, m, title in [(axes[0], base, base_name), (axes[1], best, best_name)]:
        bins = np.linspace(-0.1, 1.0, 55)
        ax.hist(m["gen"], bins, color="#55A868", alpha=0.7, density=True, label="genuine")
        ax.hist(m["imp"], bins, color="#C44E52", alpha=0.6, density=True, label="impostor")
        ax.axvline(GLOBAL_T, color="k", ls=":", lw=1.2, label=f"global t={GLOBAL_T}")
        ax.axvline(m["best_t"], color="#3b5bdb", ls="--", lw=1.4, label=f"calibrated t={m['best_t']:.2f}")
        ax.set_title(f"{title}\nEER {m['eer']*100:.1f}%  TAR@FMR1% {m['tar_at_fmr1pct']*100:.0f}%")
        ax.set_xlabel("cosine similarity"); ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    plt.suptitle("Genuine vs Impostor separation — baseline vs enhanced", fontweight="bold")
    plt.tight_layout(); plt.savefig(OUT / "acc_scores_hist.png", bbox_inches="tight", dpi=150); plt.close()


def fig_bars(all_results):
    names = list(all_results); x = np.arange(len(names)); w = 0.4
    top1 = [all_results[n]["top1"] * 100 for n in names]
    tar = [all_results[n]["tar_at_fmr1pct"] * 100 for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, top1, w, label="Top-1 identification %", color="#4C72B0")
    ax.bar(x + w/2, tar, w, label="TAR @ FMR=1%", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels([n.split(" ", 1)[0] for n in names])
    ax.set_ylabel("%"); ax.set_ylim(0, 105); ax.legend()
    ax.set_title("Accuracy by strategy (higher is better)")
    for i, (a, b) in enumerate(zip(top1, tar)):
        ax.text(i - w/2, a + 1, f"{a:.0f}", ha="center", fontsize=8)
        ax.text(i + w/2, b + 1, f"{b:.0f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "acc_strategy_bars.png", bbox_inches="tight", dpi=150); plt.close()


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    global DATA
    settings = get_settings()
    print("loading Olivetti (40 identities) ...")
    DATA = fetch_olivetti_faces(shuffle=False)

    models = {"ArcFace-MobileNet (w600k_mbf)": settings.face_recognition_model_path}
    r50 = Path("work/models/w600k_r50.onnx")
    if r50.exists():
        models["ArcFace-ResNet50 (w600k_r50)"] = r50

    all_out = {}
    for label, path in models.items():
        model = FaceEmbeddingModel(
            model_version="arcface", detection_model_path=settings.face_detection_model_path,
            recognition_model_path=path, embedding_dimensions=512,
            detection_score_threshold=settings.detection_score_threshold)
        res, embt = run_model(model, label)
        all_out[label] = {"strategies": res, "embed_seconds": embt}

    # figures from the primary (mbf) model's strategies
    primary = list(all_out)[0]
    strat = all_out[primary]["strategies"]
    fig_det(strat, primary)
    fig_bars(strat)
    base = strat["S0 baseline (mean proto)"]
    best = max(strat.values(), key=lambda m: m["tar_at_fmr1pct"])
    best_name = [n for n, m in strat.items() if m is best][0]
    fig_hist(base, best, "S0 baseline", best_name)

    # JSON (strip arrays)
    def clean(m):
        return {k: v for k, v in m.items() if k not in ("gen", "imp", "fmr", "fnmr", "ts")}
    dump = {lab: {"embed_seconds": d["embed_seconds"],
                  "strategies": {n: clean(m) for n, m in d["strategies"].items()}}
            for lab, d in all_out.items()}
    (OUT / "acc_metrics.json").write_text(json.dumps(dump, indent=2), encoding="utf-8")
    write_report(dump, primary, best_name)
    print("\nwrote acc_* figures + ACCURACY_REPORT.md to analysis/output/")


def write_report(dump, primary, best_name):
    lines = ["# Accuracy Enhancement — Recognition Model\n",
             "Protocol: Olivetti 40 identities, enrol 7 / test 3. "
             f"Global threshold {GLOBAL_T}; recalibrated threshold from DET at FMR={FMR_TARGET*100:.0f}%.\n"]
    for lab, d in dump.items():
        lines.append(f"\n## {lab}\n")
        lines.append("| Strategy | Top-1 | EER | TAR@FMR1% | Best-F1 | Best-t | F1@0.35 |")
        lines.append("|---|---|---|---|---|---|---|")
        for n, m in d["strategies"].items():
            lines.append(f"| {n} | {m['top1']*100:.1f}% | {m['eer']*100:.1f}% | "
                         f"{m['tar_at_fmr1pct']*100:.1f}% | {m['best_f1']:.3f} | "
                         f"{m['best_t']:.2f} | {m['f1_global']:.3f} |")
    lines.append(f"\n**Best strategy:** {best_name}. ")
    lines.append("Figures: `acc_det_curve.png`, `acc_scores_hist.png`, `acc_strategy_bars.png`.")
    (OUT / "ACCURACY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
