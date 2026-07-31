# Test Execution Report — Smart Attendance System

**Module:** MA981-7-FY · MSc Data Science and Its Applications · University of Essex
**Date run:** 2026-07-18
**Environment:** Windows 11 · Python 3.14.5 · pytest 9.1.1 · onnxruntime CPU
**Command:** `python -m pytest tests/ -v`

---

## 1. Automated unit / fairness suite (`tests/`)

> **UPDATE — 2026-07-31: the earlier failure was a real bug, now fixed.**
> An earlier version of this report described `test_zero_cross_person_matches`
> (17.5% cross-person false-positive rate) as an intentional demonstration of the
> thesis. That conclusion was **wrong**. Root cause below; the suite now passes.

| Test | Result | Notes |
|---|---|---|
| `TestOlivettiOverall::test_overall_f1` | ✅ PASS | End-to-end pipeline F1 meets the minimum accuracy gate. |
| `TestOlivettiOverall::test_zero_cross_person_matches` | ✅ PASS | Cross-person false-positive gate (≤ 5%) — **was 17.5%, fixed**. |
| `TestFairFaceEquity::test_per_race_detection_rate` | ⏭ SKIP | Requires the 5 GB FairFace dataset (`work/fairface`), not bundled. |
| `TestFairFaceEquity::test_per_race_f1` | ⏭ SKIP | Requires FairFace dataset. |
| `TestFairFaceEquity::test_f1_fairness_gap` | ⏭ SKIP | Requires FairFace dataset. |

**Summary:** `2 passed, 3 skipped` in ~16 s.

### Root cause of the former failure

Two defects in the face-alignment / preprocessing path
(`worker/model_adapter.py`, mirrored in `faceapi/engine.py`):

1. **Swapped eye landmarks.** `_yunet_to_arcface_kps` mapped YuNet's eye pair to
   ArcFace's template in reverse order. Because `cv2.estimateAffinePartial2D`
   models only similarity transforms (rotation, scale, translation — **no
   reflection**), that mismatch cannot be satisfied and yields a badly distorted
   112×112 crop. Every face was warped toward a similar distortion, collapsing
   inter-person separation.
2. **Wrong colour order.** ArcFace (InsightFace) is trained on RGB — its own
   loader sets `swapRB=True` — but the BGR crop was passed through unchanged.

**Measured effect** (LFW, 6 identities, real colour faces):

| Pipeline | Genuine μ | Impostor μ | Separation |
|---|---|---|---|
| Before (both defects) | 0.516 | 0.453 | 0.063 |
| + landmark order fixed | 0.615 | 0.022 | 0.593 |
| + RGB colour order fixed | **0.637** | **0.012** | **0.625** |

Mean impostor similarity fell from **0.45 to 0.012** — a ~10× improvement in
genuine/impostor separation, and the reason the false-positive gate now passes.

**Consequence for earlier results.** Any accuracy, threshold-calibration, or
fairness figure produced before this fix was measured on the broken pipeline and
must be re-derived. In particular, the earlier conclusion that "the global 0.35
threshold is hopelessly miscalibrated" was an artefact of the distorted
alignment, not a property of the model.

The three FairFace tests **skip automatically** when the dataset is absent — an
intentional design so the suite runs anywhere without a 5 GB download.

## 2. Integration / load scripts (require a running server or full dataset)

These are operational scripts, not part of the offline unit suite. They need a
live API (`python start_server.py`, port 8000) and/or external datasets, so they
were **not executed** in this offline run. They are documented here for
completeness and reproducibility:

| Script | Purpose | Requirement |
|---|---|---|
| `tests/load/locustfile.py` | HTTP load test of the ingest endpoint | Running server + `locust` |
| `tests/stress/seed_data.py` | Seed synthetic students & attendance history | Database access |
| `tests/stress/image_seed.py` | Olivetti end-to-end enrol + recognise over HTTP | Running server |
| `tests/validation/accuracy_eval.py` | Precision/recall/F1 from a predictions CSV | Predictions CSV |
| `tests/validation/worker_throughput.py` | Worker throughput via direct job processing | Database access |
| `tools/fairness_test.py`, `tools/fairface_validation.py` | FairFace fairness validation | FairFace dataset |

> The offline performance characteristics these scripts would measure are
> instead quantified reproducibly in `PERFORMANCE_REPORT.md`
> (32 images/sec sustained single-worker throughput; sub-10 ms matching at
> 200k enrolled prototypes).

## Artefacts
- `pytest_output.txt` — full verbose console log
- `pytest_junit.xml` — machine-readable JUnit results
