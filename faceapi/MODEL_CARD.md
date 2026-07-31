# Model Card — Face Recognition API

## Overview
A face **recognition service** (not a from-scratch model): it composes two
peer-reviewed pre-trained models into a calibrated, multi-prototype
identification/verification pipeline.

| Component | Model | Source | Size |
|---|---|---|---|
| Detection | YuNet (`face_detection_yunet_2023mar`) | OpenCV Zoo | 0.23 MB |
| Recognition | ArcFace MobileNet (`w600k_mbf`) | InsightFace `buffalo_sc` | 13.6 MB |
| Recognition (heavy, optional) | ArcFace ResNet-50 (`w600k_r50`) | InsightFace `buffalo_l` | 174 MB |
| Embedding | 512-d, L2-normalised; cosine similarity | — | — |

## Intended use
- 1:N **identification** and 1:1 **verification** for cooperative subjects
  (enrolment with several photos), e.g. attendance, access control, KYC-style checks.
- On-premise / edge deployment on CPU servers and ARM SBCs.

**Out of scope / not recommended:** covert surveillance, matching against
non-consenting individuals, law-enforcement identification, emotion/health
inference, or any high-stakes decision without a human in the loop.

## Performance (measured, this repo)
Controlled 40-identity protocol (Olivetti, grayscale — a **worst-case** low-quality
domain; real RGB webcam performs notably better):

| Configuration | Top-1 | EER | TAR@FMR=1% |
|---|---|---|---|
| Baseline (mean prototype) | 72.5% | 14.3% | 56.7% |
| + flip TTA | 81.7% | 11.4% | 59.2% |
| + multi-prototype (max-sim) | 80.8% | 12.2% | 71.7% |

Latency (desktop CPU): detection ~16–24 ms, embedding ~5 ms, end-to-end ~29 ms/image;
matching < 10 ms at 200k enrolled prototypes.

> **Threshold is domain-specific.** The default `0.42` must be recalibrated per
> deployment with `python -m faceapi.calibrate` (picks the operating point at a
> target false-match-rate). Grayscale/low-res domains need a higher threshold.

## Limitations & known failure modes
- **Domain shift:** accuracy drops on grayscale/low-resolution/heavily-degraded inputs;
  the larger R50 backbone was *worse* than MobileNet on such inputs in our tests.
- **Demographic bias:** face recognition exhibits accuracy differences across
  demographic groups (NIST FRVT, Grother et al. 2019). This service does **not**
  measure per-group accuracy (no labelled demographic data); operators should
  evaluate on their own population and consider per-group threshold calibration.
- **Single-face assumption:** enrol/verify/identify expect exactly one face per image.
- **No liveness / anti-spoofing:** a printed photo or screen replay can fool it; add a
  liveness check for security-sensitive use.
- **INT8 quantization** halves accuracy parity slightly (cos ≈ 0.93) and is slower on x86.

## Ethical & privacy considerations
- Face embeddings are **biometric data** — store securely, obtain informed consent,
  provide deletion (`DELETE /subjects/{id}`), and follow GDPR/BIPA-equivalent law.
- Keep a human in the loop for consequential decisions; log and audit matches.
- The bundled datasets used for evaluation (Olivetti) are research datasets, not
  representative of any real population.

## Reproducibility
- Models fetched by `python -m worker.download_models` (pinned URLs).
- Metrics reproduced by `python -m analysis.accuracy_enhancement`.
- Threshold calibrated by `python -m faceapi.calibrate`.

## License / attribution
Pre-trained weights © their respective authors (OpenCV Zoo, InsightFace) under
their licenses. This service code is provided for the MA981 dissertation; respect
academic-integrity and the upstream model licenses before production use.
