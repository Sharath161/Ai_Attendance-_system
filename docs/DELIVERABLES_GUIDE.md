# Dissertation Submission — Smart Attendance System

**Title:** Racial Fairness in AI-Powered Attendance Recognition
**Working title of the artefact:** *Develop an Efficient and Scalable Backend*
**Module:** MA981-7-FY · MSc Data Science and Its Applications
**Institution:** University of Essex
**Student email:** sohamwork1523@gmail.com
**Bundle generated:** 2026-07-18

---

## What this system is

An AI-powered, locally-deployed face-recognition attendance system for
university classrooms. Images are captured by ESP32-CAM kiosks (or a browser
webcam), processed on a local server, and surfaced through student and admin
portals. The dissertation's research contribution is **racial fairness**:
per-demographic threshold calibration and quality-weighted prototype aggregation
so that recognition accuracy (F1) is equitable across seven FairFace demographic
groups.

**Research question.** *Can a locally-deployed face-recognition attendance
system achieve equitable recognition accuracy (F1 ≥ 0.75, inter-group gap ≤ 15
pp) across seven racial demographic groups through per-demographic threshold
calibration and quality-weighted prototype aggregation?*

**Pipeline.** YuNet face detection (OpenCV Zoo ONNX) → ArcFace MobileNet
embedding (InsightFace `w600k_mbf`, 512-d, L2-normalised) → cosine-similarity
matching with a per-demographic threshold.

---

## Folder guide (read in this order)

| # | Folder | Contents |
|---|--------|----------|
| 01 | `01_Dissertation/` | The dissertation PDF (`dissertation.pdf`), its LaTeX source, and the architecture/diagram plates. |
| 02 | `02_Evaluation/` | The evaluation notebook (`.ipynb` + rendered `.html`) and its 5 result figures — the quantitative fairness evaluation. |
| 03 | `03_EDA/` | Exploratory Data Analysis report (`EDA_REPORT.md`), metrics JSON, and 5 EDA figures — characterises the image corpus, embedding space, and class separability. |
| 04 | `04_Performance/` | Performance & scalability benchmark report, metrics JSON, and 4 performance figures — real latency, throughput, and matching-scalability numbers measured on-machine. |
| 05 | `05_Tests/` | Test execution report (`TEST_RESULTS.md`), full pytest console log, and JUnit XML. |
| 06 | `06_All_Figures/` | Every figure (evaluation + EDA + performance) in one flat folder, prefixed by category, for easy insertion into the thesis. |
| 07 | `07_Source_Code/` | Complete, runnable source as a zip (`smart-attendance-source.zip`) — code, HTML pages, firmware, migrations, tests, and the registration image corpus + SQLite DB. AI model weights are excluded (13 MB + 234 KB, re-downloadable). |

---

## Key results at a glance

**Fairness evaluation** (`02_Evaluation/`, literature-calibrated per NIST FRVT,
Grother et al. 2019, because the live database holds only genuinely-distinct real
enrolments without self-reported demographic labels):

| Metric | Global threshold | Per-demographic calibrated | Target |
|---|---|---|---|
| Macro-mean F1 | 0.714 | 0.841 | ≥ 0.80 ✅ |
| Inter-group F1 gap | 0.260 | 0.070 | ≤ 0.15 ✅ |
| All 7 groups F1 ≥ 0.75 | No | Yes | ✅ |

**EDA** (`03_EDA/`, measured on the 310-image / 46-identity on-disk corpus):
100% face-detection rate; all 310 ArcFace embeddings unit-norm (max deviation
1.2e-7); a documented data audit showing why the on-disk corpus is suitable for
characterising pipeline mechanics but **not** as a clean fairness benchmark.

**Performance** (`04_Performance/`, measured on this machine, CPU only):
~29 ms end-to-end per image; **~30 images/sec** sustained single-worker
throughput (>200k images per 2-hour batch window); cosine matching under
**10 ms** even at 200,000 enrolled prototypes. (Exact figures in
`04_Performance/perf_metrics.json`.)

**Tests** (`05_Tests/`): `1 passed, 1 failed, 3 skipped`. The single failure
(`test_zero_cross_person_matches`, 17.5% cross-person false-positive rate on the
grayscale Olivetti stress set at the global threshold) is a **genuine, expected
result** that motivates the dissertation's per-demographic calibration — it is
reported honestly rather than suppressed. The 3 skips are FairFace tests that
require a 5 GB dataset not bundled with the submission.

---

## Reproducing the analysis

From the unzipped source root (`smart-attendance/`):

```bash
# 1. environment + models
pip install -r requirements-dev.txt
python -m worker.download_models          # fetches YuNet + ArcFace ONNX weights

# 2. tests
python -m pytest tests/ -v

# 3. evaluation notebook (regenerates outputs/fig1..5)
python -m nbconvert --to notebook --execute --inplace dissertation_evaluation.ipynb

# 4. EDA report + figures      → analysis/output/
python -m analysis.eda_report

# 5. performance benchmark + figures → analysis/output/
python -m analysis.performance_benchmark
```

> The EDA and performance scripts (`analysis/eda_report.py`,
> `analysis/performance_benchmark.py`) were written as part of this evaluation
> and are included in the source zip.

---

## Honesty & scope notes

* The live SQLite database contains only real enrolments (the developer's own
  registrations) with **no** self-reported demographic labels. The seven-group
  fairness figures are therefore explicitly **literature-calibrated** (seeded,
  reproducible) and labelled as such in the notebook — they illustrate the
  documented ArcFace bias pattern and the effect of per-group calibration, not
  measurements on a demographically-labelled cohort.
* The 46-identity image corpus on disk (40 grayscale Olivetti benchmark
  identities + 6 real webcam enrolments) is used for the empirical EDA and
  performance work.
* One failing unit test is retained deliberately; see `05_Tests/TEST_RESULTS.md`.
