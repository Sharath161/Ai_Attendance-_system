"""Few-shot benchmark on REAL faces (LFW) — the low-data / all-classes proof.

Protocol
--------
* LFW (funneled, colour), people with >=10 images.
* 120 people = enrolled classes; the rest = open-set distractors + s-norm cohort.
* Enrol K in {1,2,3,5} photos per person; test on the SAME held-out 5 photos of
  each person for every K (so curves are comparable).
* Strategies:  baseline (1 embedding/photo)  vs  amplified (6 aug embeddings/photo)
               each with/without s-norm score normalisation.
* Metrics:  top-1 accuracy, per-class accuracy (mean / worst / std across classes),
            open-set DIR@FAR=1% (unknown probes must be rejected).
* Speed:    end-to-end ms/img at full-res detection vs edge profile
            (detect_max_side=320).

Run:  python -m analysis.fewshot_benchmark
Outputs: analysis/output/fewshot_*.png, fewshot_metrics.json, FEWSHOT_REPORT.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from faceapi.config import get_settings
from faceapi.engine import FaceEngine, _l2
from faceapi.fewshot import augment_bank

OUT = Path("analysis/output"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})

KS = [1, 2, 3, 5]
N_TEST = 5
N_ENROLLED = 120
FAR_TARGET = 0.01
BLUE, GREEN, RED, PURPLE = "#4C72B0", "#55A868", "#C44E52", "#8172B2"


def to_bgr(img01):                       # LFW float RGB [0,1] -> uint8 BGR
    return (np.clip(img01, 0, 1) * 255).astype(np.uint8)[:, :, ::-1].copy()


def build_cache(engine, images, batch=32):
    """Per image: aligned crop -> 6 augmented embeddings. Returns [N,6,512] + ok mask."""
    crops, ok = [], np.zeros(len(images), dtype=bool)
    slots = []
    for i, im in enumerate(images):
        cs = engine.align_crops(to_bgr(im), max_faces=5)
        if cs:
            ok[i] = True
            slots.append(i)
            crops.extend(augment_bank(cs[0]["crop"]))
    E = engine.embed_crops(crops, batch=batch)          # [n_ok*6, 512]
    cache = np.zeros((len(images), 6, 512), dtype=np.float32)
    for j, i in enumerate(slots):
        cache[i] = E[j * 6:(j + 1) * 6]
    return cache, ok


def query_emb(cache_row):
    """Query embedding = flip-TTA (mean of original + flipped variants)."""
    return _l2((cache_row[0] + cache_row[1]) / 2.0)


def snorm_scores(raw, q, protos_mean, cohort):
    cq = cohort @ q
    cp = cohort @ protos_mean
    return 0.5 * ((raw - cq.mean()) / (cq.std() + 1e-6) +
                  (raw - cp.mean()) / (cp.std() + 1e-6))


def evaluate(people, cache, ok, K, amplified, use_snorm, cohort):
    """Return per-class accuracy list + genuine/impostor open-set scores."""
    # galleries
    gal = {}          # pid -> [n,512]
    gmean = {}
    for pid, idxs in people["enrolled"].items():
        rows = [cache[i] for i in idxs[:K] if ok[i]]
        if not rows:
            continue
        embs = np.concatenate([r if amplified else r[:1] for r in rows], axis=0)
        gal[pid] = embs
        gmean[pid] = _l2(embs.mean(axis=0))
    pids = list(gal)

    per_class, genuine_best = {}, []
    for pid in pids:
        test_idxs = [i for i in people["enrolled_test"][pid] if ok[i]]
        correct = 0
        for i in test_idxs:
            q = query_emb(cache[i])
            scores = {p: float(np.max(gal[p] @ q)) for p in pids}
            if use_snorm:
                scores = {p: snorm_scores(s, q, gmean[p], cohort) for p, s in scores.items()}
            best = max(scores, key=scores.get)
            genuine_best.append((scores[best], best == pid))
            correct += (best == pid)
        if test_idxs:
            per_class[pid] = correct / len(test_idxs)

    # open-set: unknown probes -> best score vs gallery (should be LOW)
    impostor_best = []
    for i in people["unknown_probes"]:
        if not ok[i]:
            continue
        q = query_emb(cache[i])
        scores = {p: float(np.max(gal[p] @ q)) for p in pids}
        if use_snorm:
            scores = {p: snorm_scores(s, q, gmean[p], cohort) for p, s in scores.items()}
        impostor_best.append(max(scores.values()))
    return per_class, genuine_best, np.array(impostor_best)


def dir_at_far(genuine_best, impostor_best, far=FAR_TARGET):
    """Open-set DIR@FAR: threshold set so far of unknowns are accepted;
    DIR = fraction of genuine probes accepted AND correctly identified."""
    if not len(impostor_best):
        return None, None
    thr = float(np.quantile(impostor_best, 1 - far))
    ok_id = sum(1 for s, correct in genuine_best if s >= thr and correct)
    return ok_id / len(genuine_best), thr


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from sklearn.datasets import fetch_lfw_people
    print("loading LFW ...")
    d = fetch_lfw_people(min_faces_per_person=10, color=True, resize=0.6,
                         slice_=None, download_if_missing=True)
    n_people = len(d.target_names)
    print(f"  {d.images.shape[0]} images, {n_people} people, frame {d.images.shape[1]}x{d.images.shape[2]}")

    # split identities
    by_pid = {}
    for i, t in enumerate(d.target):
        by_pid.setdefault(int(t), []).append(i)
    pids_sorted = sorted(by_pid, key=lambda p: -len(by_pid[p]))
    enrolled_pids = pids_sorted[:N_ENROLLED]
    distractor_pids = pids_sorted[N_ENROLLED:]

    people = {
        "enrolled": {p: by_pid[p][:max(KS)] for p in enrolled_pids},              # enrol pool (first 5)
        "enrolled_test": {p: by_pid[p][-N_TEST:] for p in enrolled_pids},         # fixed test (last 5)
        "unknown_probes": [i for p in distractor_pids for i in by_pid[p][:3]],
        "cohort_idx": [by_pid[p][3] for p in distractor_pids if len(by_pid[p]) > 3],
    }
    print(f"  enrolled classes={len(enrolled_pids)}  distractors={len(distractor_pids)} "
          f"(unknown probes={len(people['unknown_probes'])}, cohort={len(people['cohort_idx'])})")

    s = get_settings()
    engine = FaceEngine(s.detection_model_path, s.recognition_model_full,
                        s.detection_score_threshold)

    print("building embedding cache (detect+align once, 6 aug embeddings per image) ...")
    t0 = time.perf_counter()
    cache, ok = build_cache(engine, d.images)
    print(f"  cached {int(ok.sum())}/{len(ok)} images with a detected face "
          f"in {time.perf_counter()-t0:.0f}s")

    cohort = np.stack([query_emb(cache[i]) for i in people["cohort_idx"] if ok[i]])

    results = {}
    for K in KS:
        for amp in (False, True):
            for sn in (False, True):
                per_class, gen, imp = evaluate(people, cache, ok, K, amp, sn, cohort)
                accs = np.array(list(per_class.values()))
                dir1, thr = dir_at_far(gen, imp)
                key = f"K={K}|{'amp' if amp else 'base'}|{'snorm' if sn else 'raw'}"
                results[key] = {
                    "K": K, "amplified": amp, "snorm": sn,
                    "top1": float(accs.mean()),
                    "worst_class": float(accs.min()),
                    "class_std": float(accs.std()),
                    "classes_at_100": int((accs == 1.0).sum()),
                    "dir_at_far1": None if dir1 is None else float(dir1),
                    "openset_thr": thr,
                }
                r = results[key]
                print(f"  {key:<22} top1={r['top1']*100:5.1f}%  worst={r['worst_class']*100:5.1f}%  "
                      f"std={r['class_std']*100:4.1f}pp  DIR@FAR1%={r['dir_at_far1']*100:5.1f}%")

    # ── speed: full vs edge profile ──────────────────────────────────────────
    print("speed profile (end-to-end detect+embed, 30 imgs) ...")
    sample = [to_bgr(d.images[i]) for i in range(0, 300, 10)]
    prof = {}
    for label, cap in (("full", 0), ("edge(320)", 320)):
        eng = FaceEngine(s.detection_model_path, s.recognition_model_full,
                         s.detection_score_threshold, detect_max_side=cap)
        for im in sample[:3]:
            eng.detect(im)
        t0 = time.perf_counter()
        det_ok = 0
        for im in sample:
            det_ok += bool(eng.detect(im))
        ms = (time.perf_counter() - t0) / len(sample) * 1000
        prof[label] = {"ms_per_img": ms, "detected": det_ok, "of": len(sample)}
        print(f"  {label:<10} {ms:6.1f} ms/img   detected {det_ok}/{len(sample)}")

    # ── figures ──────────────────────────────────────────────────────────────
    series = {
        "baseline": ("base", "raw", RED),
        "+amplified": ("amp", "raw", BLUE),
        "+amplified +s-norm": ("amp", "snorm", GREEN),
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for name, (a, snm, col) in series.items():
        top1 = [results[f"K={k}|{a}|{snm}"]["top1"] * 100 for k in KS]
        worst = [results[f"K={k}|{a}|{snm}"]["worst_class"] * 100 for k in KS]
        dirs = [results[f"K={k}|{a}|{snm}"]["dir_at_far1"] * 100 for k in KS]
        axes[0].plot(KS, top1, "o-", color=col, lw=2, label=name)
        axes[1].plot(KS, worst, "o-", color=col, lw=2, label=name)
        axes[2].plot(KS, dirs, "o-", color=col, lw=2, label=name)
    for ax, title, yl in [
            (axes[0], "Closed-set Top-1 (mean over classes)", "accuracy (%)"),
            (axes[1], "Worst-class accuracy (fairness floor)", "accuracy (%)"),
            (axes[2], f"Open-set DIR@FAR={FAR_TARGET*100:.0f}% (unknowns rejected)", "DIR (%)")]:
        ax.set_title(title, fontsize=10); ax.set_xlabel("enrolment photos per person (K)")
        ax.set_ylabel(yl); ax.set_xticks(KS); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    plt.suptitle(f"Few-shot performance on real faces (LFW, {len(people['enrolled'])} classes, "
                 "fixed 5-photo test set)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "fewshot_curves.png", bbox_inches="tight", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = list(prof); vals = [prof[l]["ms_per_img"] for l in labels]
    bars = ax.bar(labels, vals, color=[PURPLE, GREEN], width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f} ms", ha="center", va="bottom")
    ax.set_ylabel("end-to-end ms / image")
    ax.set_title("Low-hardware profile: detection downscaling (detect_max_side=320)")
    plt.tight_layout()
    plt.savefig(OUT / "fewshot_speed_profile.png", bbox_inches="tight", dpi=150); plt.close()

    (OUT / "fewshot_metrics.json").write_text(
        json.dumps({"results": results, "speed": prof,
                    "protocol": {"classes": len(enrolled_pids), "test_per_class": N_TEST,
                                 "unknown_probes": len(people["unknown_probes"])}}, indent=2),
        encoding="utf-8")

    # report
    md = ["# Few-Shot Benchmark — real faces (LFW)\n",
          f"Protocol: {len(enrolled_pids)} enrolled classes, fixed 5-photo test set per class, "
          f"{len(people['unknown_probes'])} unknown (open-set) probes, "
          f"query uses flip-TTA. FAR target {FAR_TARGET*100:.0f}%.\n",
          "| K shots | Strategy | Top-1 | Worst class | Class std | DIR@FAR1% |",
          "|---|---|---|---|---|---|"]
    for k in KS:
        for name, (a, snm, _) in series.items():
            r = results[f"K={k}|{a}|{snm}"]
            md.append(f"| {k} | {name} | {r['top1']*100:.1f}% | {r['worst_class']*100:.1f}% | "
                      f"{r['class_std']*100:.1f}pp | {r['dir_at_far1']*100:.1f}% |")
    md += ["", "## Speed (low-hardware profile)",
           "| Profile | ms/image |", "|---|---|"]
    for l, p in prof.items():
        md.append(f"| {l} | {p['ms_per_img']:.1f} |")
    md += ["", "Figures: `fewshot_curves.png`, `fewshot_speed_profile.png`."]
    (OUT / "FEWSHOT_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print("\nwrote fewshot_* figures + FEWSHOT_REPORT.md")


if __name__ == "__main__":
    main()
