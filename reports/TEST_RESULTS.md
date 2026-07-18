# Test Execution Report — Smart Attendance System

**Module:** MA981-7-FY · MSc Data Science and Its Applications · University of Essex
**Date run:** 2026-07-18
**Environment:** Windows 11 · Python 3.14.5 · pytest 9.1.1 · onnxruntime CPU
**Command:** `python -m pytest tests/ -v`

---

## 1. Automated unit / fairness suite (`tests/`)

| Test | Result | Notes |
|---|---|---|
| `TestOlivettiOverall::test_overall_f1` | ✅ PASS | End-to-end pipeline F1 on the Olivetti benchmark meets the minimum accuracy gate. |
| `TestOlivettiOverall::test_zero_cross_person_matches` | ❌ FAIL | Cross-person false-positive rate = **17.5%** at the global threshold (gate ≤ 5%). |
| `TestFairFaceEquity::test_per_race_detection_rate` | ⏭ SKIP | Requires the 5 GB FairFace dataset (`work/fairface`), not bundled. |
| `TestFairFaceEquity::test_per_race_f1` | ⏭ SKIP | Requires FairFace dataset. |
| `TestFairFaceEquity::test_f1_fairness_gap` | ⏭ SKIP | Requires FairFace dataset. |

**Summary:** `1 failed, 1 passed, 3 skipped` in ~10 s.

### Why the FAIL is a genuine, expected result (not a defect to hide)

`test_zero_cross_person_matches` runs on the **Olivetti** faces dataset —
grayscale 64×64 images upscaled to the pipeline input. It is deliberately a hard
false-positive gate. The observed 17.5% cross-person false-positive rate at the
single global cosine threshold (0.35) is exactly the phenomenon the dissertation
investigates: **a single global threshold is unsafe**, and it is worse on
harder / out-of-distribution image domains. This mirrors the NIST FRVT finding
(Grother et al., 2019) that false-match behaviour varies sharply by input
population, motivating the per-demographic threshold calibration that is the
project's core contribution. The failing assertion is therefore reported
faithfully rather than relaxed to make the suite green.

The EDA (see `EDA_REPORT.md`, Figure 5) independently confirms the mechanism:
on the grayscale Olivetti domain the genuine and impostor cosine-similarity
distributions nearly overlap (μ 0.72 vs 0.78), so no fixed threshold cleanly
separates them.

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
