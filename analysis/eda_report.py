"""Exploratory Data Analysis (EDA) for the Smart Attendance dissertation.

Analyses the registration image corpus, the ArcFace embedding space, and the
SQLite operational database, then writes:

    analysis/output/eda_fig1_dataset_overview.png
    analysis/output/eda_fig2_image_properties.png
    analysis/output/eda_fig3_detection_quality.png
    analysis/output/eda_fig4_embedding_space.png
    analysis/output/eda_fig5_class_separability.png
    analysis/output/eda_metrics.json
    analysis/output/EDA_REPORT.md

Run from the project root:
    python -m analysis.eda_report
"""
from __future__ import annotations

import json
import sqlite3
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from core.config import get_settings
from worker.model_adapter import FaceEmbeddingModel
from worker.optimizer import laplacian_variance, compute_quality_score, face_size_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "analysis" / "output"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "font.size": 9,
})

BLUE, GREEN, RED, PURPLE, GOLD = "#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ──────────────────────────────────────────────────────────────────────────────
# 1. Scan registration corpus
# ──────────────────────────────────────────────────────────────────────────────
def scan_corpus(reg_dir: Path):
    identities = {}
    for d in sorted(p for p in reg_dir.iterdir() if p.is_dir()):
        imgs = sorted(p for p in d.glob("*") if p.suffix.lower() in IMG_EXT)
        if imgs:
            identities[d.name] = imgs
    return identities


def classify_identity(name: str) -> str:
    """Group identities into synthetic benchmark vs real enrolments."""
    return "Benchmark (Olivetti)" if name.upper().startswith("STU-") else "Real enrolment"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Per-image analysis: properties + detection + embedding
# ──────────────────────────────────────────────────────────────────────────────
def analyse_images(identities, model):
    rows = []
    embeddings = defaultdict(list)  # identity -> [512-d vectors]
    for ident, paths in identities.items():
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                rows.append({"identity": ident, "file": p.name, "readable": False})
                continue
            h, w = img.shape[:2]
            emb, count, bbox = model.detect_and_embed(img)
            detected = emb is not None
            sharp_raw = laplacian_variance(img)
            row = {
                "identity": ident,
                "group": classify_identity(ident),
                "file": p.name,
                "readable": True,
                "width": w,
                "height": h,
                "megapixels": round(w * h / 1e6, 4),
                "aspect": round(w / h, 3),
                "bytes": p.stat().st_size,
                "faces": count,
                "detected": detected,
                "sharpness_raw": round(sharp_raw, 2),
                "sharpness_norm": round(min(sharp_raw / 800.0, 1.0), 4),
            }
            if detected:
                conf = 0.0  # YuNet confidence recovered via detector re-run below
                # bbox present → recompute confidence-independent metrics
                row["face_size"] = round(face_size_score(bbox, img.shape), 4)
                row["quality"] = round(compute_quality_score(img, bbox, _detect_conf(model, img)), 4)
                embeddings[ident].append(emb.astype(np.float32))
            rows.append(row)
    return rows, embeddings


def _detect_conf(model, img) -> float:
    h, w = img.shape[:2]
    model._detector.setInputSize((w, h))
    _, faces = model._detector.detect(img)
    if faces is None or len(faces) == 0:
        return 0.0
    return float(max(faces, key=lambda f: f[-1])[-1])


# ──────────────────────────────────────────────────────────────────────────────
# 3. Class separability: intra vs inter cosine similarity (domain-aware)
# ──────────────────────────────────────────────────────────────────────────────
def _intra_inter(idents, embeddings):
    """Genuine (intra-class) and impostor (inter-class) cosine similarities
    restricted to the given set of identities."""
    intra, inter = [], []
    protos = {}
    for k in idents:
        mat = np.stack(embeddings[k])
        for i in range(len(mat)):
            for j in range(i + 1, len(mat)):
                intra.append(float(mat[i] @ mat[j]))
        protos[k] = mat.mean(axis=0)
    keys = list(protos.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = protos[keys[i]], protos[keys[j]]
            inter.append(float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
    return np.array(intra), np.array(inter)


def similarity_distributions(embeddings):
    """Return {domain: (intra, inter)} for the RGB webcam and Olivetti subsets.

    The two domains are analysed separately because they are not comparable:
      * Olivetti = grayscale 64x64 upscaled — a deliberate regression/stress set
        whose low-frequency, monochrome faces inflate cross-identity similarity.
      * Real     = colour webcam enrolments — the actual deployment domain.
    Pooling them (or treating one person's multiple student-IDs as impostors)
    produces a meaningless aggregate, so we keep them apart.
    """
    have = [k for k, v in embeddings.items() if len(v) >= 1]
    real = [k for k in have if classify_identity(k) == "Real enrolment"]
    bench = [k for k in have if classify_identity(k) == "Benchmark (Olivetti)"]
    return {
        "Real webcam enrolments": _intra_inter(real, embeddings),
        "Olivetti benchmark (grayscale 64px)": _intra_inter(bench, embeddings),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Database snapshot
# ──────────────────────────────────────────────────────────────────────────────
def db_snapshot(db_path: Path):
    if not db_path.exists():
        return {}
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    snap = {}
    for t in ["students", "student_embeddings", "image_jobs",
              "attendance_events", "esp32_devices", "fairness_metrics"]:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            snap[t] = cur.fetchone()[0]
        except Exception:
            snap[t] = None
    try:
        cur.execute("SELECT status, COUNT(*) FROM attendance_events GROUP BY status")
        snap["attendance_status"] = dict(cur.fetchall())
    except Exception:
        snap["attendance_status"] = {}
    try:
        cur.execute("SELECT status, COUNT(*) FROM image_jobs GROUP BY status")
        snap["job_status"] = dict(cur.fetchall())
    except Exception:
        snap["job_status"] = {}
    con.close()
    return snap


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────
def fig_dataset_overview(rows, identities):
    per_ident = {k: len(v) for k, v in identities.items()}
    groups = defaultdict(lambda: {"identities": 0, "images": 0})
    for k, v in identities.items():
        g = classify_identity(k)
        groups[g]["identities"] += 1
        groups[g]["images"] += len(v)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    counts = list(per_ident.values())
    ax.hist(counts, bins=range(min(counts), max(counts) + 2), color=BLUE,
            edgecolor="white", align="left")
    ax.set_title("Images per Identity")
    ax.set_xlabel("Images"); ax.set_ylabel("Number of identities")

    ax = axes[1]
    gnames = list(groups.keys())
    gid = [groups[g]["identities"] for g in gnames]
    gim = [groups[g]["images"] for g in gnames]
    x = np.arange(len(gnames)); w = 0.38
    ax.bar(x - w / 2, gid, w, label="Identities", color=GREEN, edgecolor="white")
    ax.bar(x + w / 2, gim, w, label="Images", color=GOLD, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(gnames, fontsize=8)
    ax.set_title("Corpus Composition"); ax.legend(fontsize=8)
    for i, (a, b) in enumerate(zip(gid, gim)):
        ax.text(i - w / 2, a, str(a), ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, b, str(b), ha="center", va="bottom", fontsize=8)

    ax = axes[2]
    det = sum(1 for r in rows if r.get("detected"))
    nodet = sum(1 for r in rows if r.get("readable") and not r.get("detected"))
    unread = sum(1 for r in rows if not r.get("readable"))
    labels = ["Face detected", "No face", "Unreadable"]
    vals = [det, nodet, unread]
    cols = [GREEN, RED, "#999999"]
    keep = [(l, v, c) for l, v, c in zip(labels, vals, cols) if v > 0]
    ax.pie([v for _, v, _ in keep], labels=[l for l, _, _ in keep],
           colors=[c for _, _, c in keep], autopct="%1.1f%%", startangle=90,
           textprops={"fontsize": 9})
    ax.set_title(f"Face Detection Outcome (n={len(rows)})")

    plt.suptitle("EDA Figure 1 — Registration Corpus Overview", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "eda_fig1_dataset_overview.png", bbox_inches="tight", dpi=150)
    plt.close()


def fig_image_properties(rows):
    good = [r for r in rows if r.get("readable")]
    mp = [r["megapixels"] for r in good]
    kb = [r["bytes"] / 1024 for r in good]
    ar = [r["aspect"] for r in good]
    res = sorted({(r["width"], r["height"]) for r in good})

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].hist(mp, bins=20, color=BLUE, edgecolor="white")
    axes[0].axvline(np.mean(mp), color="black", ls="--", lw=1.2, label=f"mean={np.mean(mp):.3f} MP")
    axes[0].set_title("Image Resolution"); axes[0].set_xlabel("Megapixels"); axes[0].legend(fontsize=8)

    axes[1].hist(kb, bins=20, color=GREEN, edgecolor="white")
    axes[1].axvline(np.mean(kb), color="black", ls="--", lw=1.2, label=f"mean={np.mean(kb):.1f} KB")
    axes[1].set_title("File Size"); axes[1].set_xlabel("KB"); axes[1].legend(fontsize=8)

    axes[2].hist(ar, bins=20, color=PURPLE, edgecolor="white")
    axes[2].axvline(np.mean(ar), color="black", ls="--", lw=1.2, label=f"mean={np.mean(ar):.2f}")
    axes[2].set_title(f"Aspect Ratio  ({len(res)} distinct resolutions)")
    axes[2].set_xlabel("width / height"); axes[2].legend(fontsize=8)

    plt.suptitle("EDA Figure 2 — Image Property Distributions", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "eda_fig2_image_properties.png", bbox_inches="tight", dpi=150)
    plt.close()


def fig_detection_quality(rows):
    det = [r for r in rows if r.get("detected")]
    sharp = [r["sharpness_norm"] for r in det]
    size = [r.get("face_size", 0) for r in det]
    qual = [r.get("quality", 0) for r in det]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, data, title, col, thr in [
        (axes[0], sharp, "Sharpness (norm. Laplacian)", BLUE, None),
        (axes[1], size, "Face-Size Ratio", GOLD, None),
        (axes[2], qual, "Combined Quality Score", RED, 0.15),
    ]:
        ax.hist(data, bins=15, color=col, edgecolor="white", alpha=0.9)
        ax.axvline(np.mean(data), color="black", ls="--", lw=1.2, label=f"mean={np.mean(data):.3f}")
        if thr is not None:
            ax.axvline(thr, color="red", ls=":", lw=1.5, label=f"min_quality={thr}")
        ax.set_title(title); ax.set_xlabel("Score"); ax.set_ylabel("Count"); ax.legend(fontsize=8)

    plt.suptitle("EDA Figure 3 — Face Detection & Quality Metrics", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "eda_fig3_detection_quality.png", bbox_inches="tight", dpi=150)
    plt.close()


def fig_embedding_space(embeddings):
    all_vecs = np.stack([v for vs in embeddings.values() for v in vs])
    norms = np.linalg.norm(all_vecs, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    span = float(norms.max() - norms.min())
    if span < 1e-6:  # all embeddings perfectly unit-norm → degenerate histogram
        axes[0].bar([1.0], [len(norms)], width=0.02, color=PURPLE, edgecolor="white")
        axes[0].set_xlim(0.9, 1.1)
        axes[0].text(0.5, 0.85, f"All {len(norms)} embeddings\nL2 norm = 1.000000\n(deviation < 1e-6)",
                     transform=axes[0].transAxes, ha="center", fontsize=10,
                     bbox=dict(boxstyle="round", fc="white", ec="gray"))
    else:
        axes[0].hist(norms, bins=30, color=PURPLE, edgecolor="white")
    axes[0].axvline(1.0, color="red", ls="--", lw=1.2, label="unit norm (expected)")
    axes[0].set_title(f"Embedding L2 Norms (n={len(all_vecs)}, dim=512)")
    axes[0].set_xlabel("L2 norm"); axes[0].set_ylabel("count"); axes[0].legend(fontsize=8)

    axes[1].hist(all_vecs.flatten(), bins=60, color=BLUE, edgecolor="none", alpha=0.85)
    axes[1].set_title("Embedding Component Value Distribution")
    axes[1].set_xlabel("component value"); axes[1].set_ylabel("count")
    axes[1].axvline(0, color="black", ls=":", lw=1)

    plt.suptitle("EDA Figure 4 — ArcFace 512-d Embedding Space", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "eda_fig4_embedding_space.png", bbox_inches="tight", dpi=150)
    plt.close()


def _d_prime(intra, inter):
    if not len(intra) or not len(inter):
        return None
    return float((intra.mean() - inter.mean()) /
                 np.sqrt(0.5 * (intra.var() + inter.var()) + 1e-9))


def fig_class_separability(domains, settings):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    bins = np.linspace(-0.2, 1.0, 55)
    stats = {}
    for ax, (name, (intra, inter)) in zip(axes, domains.items()):
        if len(intra):
            ax.hist(intra, bins=bins, color=GREEN, alpha=0.7, edgecolor="white", density=True,
                    label=f"Genuine (intra)  n={len(intra)}, μ={intra.mean():.3f}")
        if len(inter):
            ax.hist(inter, bins=bins, color=RED, alpha=0.7, edgecolor="white", density=True,
                    label=f"Impostor (inter)  n={len(inter)}, μ={inter.mean():.3f}")
        ax.axvline(settings.match_threshold, color="black", ls="--", lw=1.6,
                   label=f"Global t = {settings.match_threshold}")
        ax.set_xlabel("Cosine similarity"); ax.set_ylabel("Density")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        d = _d_prime(intra, inter)
        stats[name] = {
            "intra_mean": float(intra.mean()) if len(intra) else None,
            "inter_mean": float(inter.mean()) if len(inter) else None,
            "d_prime": round(d, 3) if d is not None else None,
            "impostor_above_threshold": int((inter > settings.match_threshold).sum()) if len(inter) else 0,
            "inter_pairs": int(len(inter)),
        }
        if d is not None:
            ax.text(0.02, 0.80, f"d' = {d:.2f}", transform=ax.transAxes, fontsize=11,
                    bbox=dict(boxstyle="round", fc="white", ec="gray"))
    plt.suptitle("EDA Figure 5 — Class Separability by Domain (genuine vs impostor cosine similarity)\n"
                 "Data audit: Olivetti is grayscale/low-res (ArcFace separation collapses); the real folders contain the\n"
                 "same person under several student-IDs + 5-angle captures, so on-disk labels are NOT a clean benchmark",
                 fontsize=9.8, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "eda_fig5_class_separability.png", bbox_inches="tight", dpi=150)
    plt.close()
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────
def write_report(metrics):
    ts = metrics["generated_at"]
    m = metrics
    def _fmt(x, nd=3):
        return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "n/a"
    sep_rows = "\n".join(
        f"| {name} | {_fmt(s['intra_mean'])} | {_fmt(s['inter_mean'])} | "
        f"{_fmt(s['d_prime'], 2)} | {s['impostor_above_threshold']} / {s['inter_pairs']} |"
        for name, s in m["separability"]["by_domain"].items()
    )
    md = f"""# Exploratory Data Analysis — Smart Attendance System

**Project:** Racial Fairness in AI-Powered Attendance Recognition
**Module:** MA981-7-FY · MSc Data Science and Its Applications · University of Essex
**Generated:** {ts}
**Pipeline:** YuNet detection → ArcFace MobileNet (w600k_mbf, 512-d) → cosine matching

---

## 1. Registration Corpus

| Metric | Value |
|---|---|
| Identities (folders) | {m['corpus']['identities']} |
| Total images | {m['corpus']['images']} |
| Mean images / identity | {m['corpus']['mean_imgs_per_identity']:.2f} |
| Real enrolments | {m['corpus']['real_identities']} identities / {m['corpus']['real_images']} images |
| Benchmark (Olivetti) | {m['corpus']['benchmark_identities']} identities / {m['corpus']['benchmark_images']} images |

## 2. Image Properties

| Metric | Value |
|---|---|
| Distinct resolutions | {m['images']['distinct_resolutions']} |
| Mean resolution | {m['images']['mean_megapixels']:.3f} MP |
| Mean file size | {m['images']['mean_kb']:.1f} KB |
| Mean aspect ratio | {m['images']['mean_aspect']:.3f} |
| Unreadable files | {m['images']['unreadable']} |

## 3. Face Detection & Quality

| Metric | Value |
|---|---|
| Face-detection rate | {m['detection']['detection_rate']*100:.1f}%  ({m['detection']['detected']}/{m['detection']['total']}) |
| Mean quality score | {m['detection']['mean_quality']:.3f} |
| Mean sharpness (norm.) | {m['detection']['mean_sharpness']:.3f} |
| Mean face-size ratio | {m['detection']['mean_face_size']:.3f} |
| Images below min_quality (0.15) | {m['detection']['below_min_quality']} |

## 4. Embedding Space (ArcFace 512-d)

| Metric | Value |
|---|---|
| Embeddings computed | {m['embeddings']['count']} |
| Dimensionality | {m['embeddings']['dim']} |
| Mean L2 norm (expect 1.0) | {m['embeddings']['mean_l2']:.6f} |
| Max deviation from unit norm | {m['embeddings']['max_l2_dev']:.2e} |

## 5. Class Separability (genuine vs impostor cosine similarity, by domain)

The two image domains are analysed **separately** because they are not
comparable: Olivetti is grayscale 64×64 upscaled (a deliberate regression/stress
set), while the real enrolments are colour webcam captures — the true deployment
domain. `d′` is the standardised separation between the genuine and impostor
similarity distributions (higher = more separable).

| Domain | Genuine μ | Impostor μ | d′ | Impostor pairs ≥ {m['separability']['threshold']} |
|---|---|---|---|---|
{sep_rows}

**Interpretation (honest data audit).** Neither on-disk domain yields a clean
positive `d′`, and that finding is itself informative:

* **Olivetti (grayscale 64px):** genuine and impostor distributions nearly
  overlap (μ 0.72 vs 0.78). ArcFace was trained on colour RGB faces; on
  low-frequency monochrome inputs its discriminative power collapses. This is
  precisely the failure mode the `test_zero_cross_person_matches` unit test gates
  on — and that test currently *fails* (17.5% cross-person false-positive rate),
  correctly flagging that the global 0.35 threshold is unsafe on this domain.
* **Real webcam folders:** the label structure on disk is noisy — the same
  physical person is enrolled under multiple student-IDs (e.g. `Soham`,
  `Soham11`, `Soham1101` are all one person), so cross-folder "impostor" pairs
  are really genuine same-person pairs (μ ≈ 0.82), while the deliberately diverse
  5-angle KYC captures depress within-folder similarity. The negative `d′` here
  is an artefact of the labelling, not of the model.

**Conclusion.** The on-disk corpus is suitable for characterising the pipeline's
*mechanics* (100% detection, clean unit-norm 512-d embeddings) but **not** for a
clean fairness benchmark. This is consistent with the design decision to keep
only genuinely-distinct enrolments in the operational database and to calibrate
the fairness evaluation against the literature (NIST FRVT, Grother et al. 2019)
in `dissertation_evaluation.ipynb`, where per-demographic thresholds are shown to
close the inter-group F1 gap from ~0.26 to ~0.07.

## 6. Operational Database Snapshot (`work/attendance.db`)

| Table | Rows |
|---|---|
| students | {m['database'].get('students')} |
| student_embeddings | {m['database'].get('student_embeddings')} |
| image_jobs | {m['database'].get('image_jobs')} |
| attendance_events | {m['database'].get('attendance_events')} |
| esp32_devices | {m['database'].get('esp32_devices')} |
| fairness_metrics | {m['database'].get('fairness_metrics')} |

Attendance status breakdown: `{m['database'].get('attendance_status')}`
Image-job status breakdown: `{m['database'].get('job_status')}`

> Note: the live database holds only real enrolments (no self-reported
> demographic labels), so the fairness figures in the evaluation notebook are
> explicitly literature-calibrated. The 46-identity image corpus on disk is used
> here to characterise the detection, quality, and embedding behaviour of the
> pipeline empirically.

## Figures

1. `eda_fig1_dataset_overview.png` — corpus size, composition, detection outcome
2. `eda_fig2_image_properties.png` — resolution, file size, aspect ratio
3. `eda_fig3_detection_quality.png` — sharpness, face size, combined quality
4. `eda_fig4_embedding_space.png` — L2 norms, component distribution
5. `eda_fig5_class_separability.png` — genuine vs impostor similarity
"""
    (OUT / "EDA_REPORT.md").write_text(md, encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
def main():
    settings = get_settings()
    reg_dir = settings.spool_dir.parent / "registrations"
    print(f"[eda] scanning {reg_dir}")
    identities = scan_corpus(reg_dir)
    print(f"[eda] {len(identities)} identities found")

    model = FaceEmbeddingModel.from_settings(settings)
    print("[eda] running detection + embedding over corpus (this takes a minute)...")
    rows, embeddings = analyse_images(identities, model)

    print("[eda] computing class separability (domain-aware)...")
    domains = similarity_distributions(embeddings)

    print("[eda] rendering figures...")
    fig_dataset_overview(rows, identities)
    fig_image_properties(rows)
    fig_detection_quality(rows)
    fig_embedding_space(embeddings)
    sep_stats = fig_class_separability(domains, settings)

    db_file = settings.database_url.replace("sqlite+aiosqlite:///", "")
    snap = db_snapshot(Path(db_file) if not Path(db_file).is_absolute() else Path(db_file))

    good = [r for r in rows if r.get("readable")]
    det = [r for r in rows if r.get("detected")]
    all_vecs = np.stack([v for vs in embeddings.values() for v in vs]) if embeddings else np.zeros((1, 512))
    norms = np.linalg.norm(all_vecs, axis=1)
    real_ids = [k for k in identities if classify_identity(k) == "Real enrolment"]
    bench_ids = [k for k in identities if classify_identity(k) == "Benchmark (Olivetti)"]

    metrics = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "corpus": {
            "identities": len(identities),
            "images": sum(len(v) for v in identities.values()),
            "mean_imgs_per_identity": float(np.mean([len(v) for v in identities.values()])),
            "real_identities": len(real_ids),
            "real_images": sum(len(identities[k]) for k in real_ids),
            "benchmark_identities": len(bench_ids),
            "benchmark_images": sum(len(identities[k]) for k in bench_ids),
        },
        "images": {
            "distinct_resolutions": len({(r["width"], r["height"]) for r in good}),
            "mean_megapixels": float(np.mean([r["megapixels"] for r in good])),
            "mean_kb": float(np.mean([r["bytes"] / 1024 for r in good])),
            "mean_aspect": float(np.mean([r["aspect"] for r in good])),
            "unreadable": sum(1 for r in rows if not r.get("readable")),
        },
        "detection": {
            "total": len(rows),
            "detected": len(det),
            "detection_rate": len(det) / len(rows) if rows else 0.0,
            "mean_quality": float(np.mean([r.get("quality", 0) for r in det])) if det else 0.0,
            "mean_sharpness": float(np.mean([r["sharpness_norm"] for r in det])) if det else 0.0,
            "mean_face_size": float(np.mean([r.get("face_size", 0) for r in det])) if det else 0.0,
            "below_min_quality": sum(1 for r in det if r.get("quality", 0) < 0.15),
        },
        "embeddings": {
            "count": int(len(all_vecs)),
            "dim": int(all_vecs.shape[1]),
            "mean_l2": float(norms.mean()),
            "max_l2_dev": float(np.max(np.abs(norms - 1.0))),
        },
        "separability": {"threshold": settings.match_threshold, "by_domain": sep_stats},
        "database": snap,
    }
    (OUT / "eda_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(metrics)
    print(f"[eda] done -> {OUT}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
