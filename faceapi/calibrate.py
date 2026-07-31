"""Calibrate the match threshold for YOUR domain (optimize + calibrate).

Point it at a labelled folder (one subfolder of photos per person). It builds
multi-prototype galleries, computes genuine vs impostor cosine scores via
leave-one-out, and picks the threshold at a target false-match-rate (default 1%).
Prints the value to put in FACEAPI_MATCH_THRESHOLD.

    python -m faceapi.calibrate --dir work/my_people --fmr 0.01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from faceapi.config import get_settings
from faceapi.engine import FaceEngine

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="work/my_people")
    ap.add_argument("--fmr", type=float, default=0.01, help="target false-match-rate")
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    s = get_settings()
    engine = FaceEngine(s.detection_model_path, s.recognition_model(),
                        s.detection_score_threshold, s.intra_op_threads, enable_tta=args.tta)

    root = Path(args.dir)
    people = [d for d in sorted(root.iterdir()) if d.is_dir()] if root.exists() else []
    embset = {}
    for d in people:
        vs = []
        for p in sorted(d.glob("*")):
            if p.suffix.lower() not in IMG_EXT:
                continue
            im = cv2.imread(str(p))
            if im is None:
                continue
            dets = engine.detect(im, embed=True, max_faces=10)
            if len(dets) == 1 and dets[0].embedding is not None:
                vs.append(dets[0].embedding)
        if len(vs) >= 2:
            embset[d.name] = np.stack(vs)
    if len(embset) < 2:
        print(f"Need >=2 people with >=2 detectable photos each in {root}. Found {len(embset)}.")
        sys.exit(1)

    # leave-one-out genuine (query vs rest of own) + impostor (query vs other people)
    genuine, impostor = [], []
    ids = list(embset)
    for sid, mat in embset.items():
        for i in range(len(mat)):
            q = mat[i]
            rest = np.delete(mat, i, axis=0)
            genuine.append(float(np.max(rest @ q)))
            for other in ids:
                if other == sid:
                    continue
                impostor.append(float(np.max(embset[other] @ q)))
    gen, imp = np.array(genuine), np.array(impostor)

    ts = np.linspace(0, 1, 1001)
    fmr = np.array([(imp >= t).mean() for t in ts])
    fnmr = np.array([(gen < t).mean() for t in ts])
    eer_i = int(np.argmin(np.abs(fmr - fnmr)))
    ok = np.where(fmr <= args.fmr)[0]
    t_star = float(ts[ok[0]]) if len(ok) else 1.0
    tar = 1 - (gen < t_star).mean()

    print("=" * 56)
    print("  THRESHOLD CALIBRATION")
    print("=" * 56)
    print(f"  people={len(embset)}  genuine pairs={len(gen)}  impostor pairs={len(imp)}")
    print(f"  genuine  cosine: mean={gen.mean():.3f}  min={gen.min():.3f}")
    print(f"  impostor cosine: mean={imp.mean():.3f}  max={imp.max():.3f}")
    print(f"  EER            : {(fmr[eer_i]+fnmr[eer_i])/2*100:.1f}%  (t={ts[eer_i]:.3f})")
    print(f"  threshold @ FMR={args.fmr*100:.0f}%: {t_star:.3f}   (TAR {tar*100:.1f}%)")
    print("=" * 56)
    print(f"  Set:  FACEAPI_MATCH_THRESHOLD={t_star:.3f}"
          + ("   FACEAPI_ENABLE_TTA=true" if args.tta else ""))


if __name__ == "__main__":
    main()
