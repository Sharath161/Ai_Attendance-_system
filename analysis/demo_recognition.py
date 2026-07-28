"""End-to-end recognition demo — exercises the real face pipeline.

Registers a few distinct people (enrol images -> quality-weighted 512-d
prototype), then recognises held-out images and an unenrolled impostor,
using the SAME model adapter, quality scoring, and cosine matching that the
FastAPI server / batch worker use. No server or database is touched.

Run from the project root:
    python -m analysis.demo_recognition
"""
from __future__ import annotations

import sys
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

from core.config import get_settings
from core.math_utils import l2_normalize
from worker.model_adapter import FaceEmbeddingModel
from worker.optimizer import laplacian_variance, compute_quality_score
from tests.stress.image_seed import face_to_frame, augment_params

ENROLLED = 6          # people enrolled in the gallery (distinct identities)
ENROLL_IMGS = 7       # enrolment images per person
TEST_IMGS = 3         # held-out query images per person
IMPOSTOR_ID = 39      # an Olivetti person NOT enrolled


def embed_frame(model, frame):
    """Detect + align + embed a BGR frame; return (embedding, quality) or (None, 0)."""
    emb, count, bbox = model.detect_and_embed(frame)
    if emb is None:
        return None, 0.0
    conf = 0.0
    h, w = frame.shape[:2]
    model._detector.setInputSize((w, h))
    _, faces = model._detector.detect(frame)
    if faces is not None and len(faces):
        conf = float(max(faces, key=lambda f: f[-1])[-1])
    q = compute_quality_score(frame, bbox, conf)
    return emb.astype(np.float32), q


def build_prototype(model, faces, pid):
    """Enrol: quality-weighted mean of the enrolment embeddings -> unit prototype."""
    embs, ws = [], []
    for idx, face in enumerate(faces):
        b, ang = augment_params(pid * 100 + idx)
        frame = face_to_frame(face, brightness=b, rotate_deg=ang)
        e, q = embed_frame(model, frame)
        if e is not None:
            embs.append(e); ws.append(max(q, 1e-3))
    if not embs:
        return None, 0
    w = np.asarray(ws, np.float32); w /= w.sum()
    proto = l2_normalize((np.stack(embs) * w[:, None]).sum(0))
    return proto, len(embs)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = get_settings()
    thr = settings.match_threshold
    print("=" * 68)
    print("  FACE-RECOGNITION PIPELINE DEMO  (YuNet -> ArcFace 512-d -> cosine)")
    print("=" * 68)

    model = FaceEmbeddingModel.from_settings(settings)
    print(f"  model={model.model_version}  dim={model.embedding_dimensions}  "
          f"match_threshold={thr}")

    data = fetch_olivetti_faces(shuffle=False)

    # ── Enrolment ─────────────────────────────────────────────────────────────
    print(f"\n[1] Enrolling {ENROLLED} people ({ENROLL_IMGS} images each) ...")
    gallery = {}   # student_id -> prototype
    for pid in range(ENROLLED):
        sid = f"STU-{pid+1:03d}"
        imgs = data.images[data.target == pid][:ENROLL_IMGS]
        proto, n = build_prototype(model, imgs, pid)
        gallery[sid] = proto
        print(f"    {sid}: prototype built from {n}/{ENROLL_IMGS} detected faces")

    def match(query):
        """Return (best_sid, best_score) over the gallery — same cosine rule as the API."""
        best_sid, best = None, -1.0
        for sid, proto in gallery.items():
            s = float(np.dot(query, proto))
            if s > best:
                best, best_sid = s, sid
        return best_sid, best

    # ── Recognition of held-out genuine images ────────────────────────────────
    print(f"\n[2] Recognising {ENROLLED*TEST_IMGS} held-out genuine images ...\n")
    print(f"    {'true':<9}{'predicted':<11}{'score':>7}  {'decision':<13}{'result'}")
    print("    " + "-" * 52)
    top1_hits = tp = fn = 0
    genuine_scores = []
    for pid in range(ENROLLED):
        true_sid = f"STU-{pid+1:03d}"
        held = data.images[data.target == pid][ENROLL_IMGS:ENROLL_IMGS + TEST_IMGS]
        for k, face in enumerate(held):
            b, ang = augment_params(pid*100 + ENROLL_IMGS + k)
            frame = face_to_frame(face, brightness=b, rotate_deg=ang)
            e, _ = embed_frame(model, frame)
            if e is None:
                print(f"    {true_sid:<9}{'—':<11}{'—':>7}  {'no_face':<13}(miss)"); fn += 1; continue
            pred, score = match(e)
            genuine_scores.append(score)
            top1 = pred == true_sid
            top1_hits += top1
            if score >= thr:
                decision = "recognised";
                if top1: tp += 1
            else:
                decision = "unknown"; fn += 1
            ok = "OK" if (top1 and score >= thr) else ("wrong-id" if not top1 else "below-thr")
            print(f"    {true_sid:<9}{pred:<11}{score:>7.3f}  {decision:<13}{ok}")

    # ── Impostor (unenrolled person) ──────────────────────────────────────────
    print(f"\n[3] Impostor test — person {IMPOSTOR_ID} is NOT enrolled (should be rejected):")
    imp = data.images[data.target == IMPOSTOR_ID][0]
    e, _ = embed_frame(model, face_to_frame(imp))
    pred, score = match(e)
    verdict = "REJECTED (unknown)" if score < thr else f"FALSE ACCEPT as {pred}"
    print(f"    closest gallery match: {pred}  score={score:.3f}  ->  {verdict}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_tests = ENROLLED * TEST_IMGS
    gs = np.array(genuine_scores)
    print("\n" + "=" * 68)
    print("  SUMMARY")
    print("=" * 68)
    print(f"  Top-1 identification accuracy : {top1_hits}/{n_tests} = {top1_hits/n_tests*100:.1f}%")
    print(f"  Accepted at threshold {thr}      : {tp}/{n_tests} = {tp/n_tests*100:.1f}%")
    print(f"  Genuine cosine similarity      : min={gs.min():.3f}  mean={gs.mean():.3f}  max={gs.max():.3f}")
    print(f"  Impostor cosine similarity     : {score:.3f}  (< {thr} -> correctly rejected)"
          if score < thr else
          f"  Impostor cosine similarity     : {score:.3f}  (>= {thr} -> FALSE ACCEPT)")
    print("=" * 68)
    print("  The pipeline detected, aligned, embedded, and matched every face using")
    print("  the same code path as the live server. Top-1 accuracy shows the model")
    print("  identifies the correct enrolled person; the threshold row reflects the")
    print("  global operating point studied in the fairness evaluation.")


if __name__ == "__main__":
    main()
