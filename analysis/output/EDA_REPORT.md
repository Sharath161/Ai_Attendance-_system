# Exploratory Data Analysis — Smart Attendance System

**Project:** Racial Fairness in AI-Powered Attendance Recognition
**Module:** MA981-7-FY · MSc Data Science and Its Applications · University of Essex
**Generated:** 2026-07-18 15:24 UTC
**Pipeline:** YuNet detection → ArcFace MobileNet (w600k_mbf, 512-d) → cosine matching

---

## 1. Registration Corpus

| Metric | Value |
|---|---|
| Identities (folders) | 46 |
| Total images | 310 |
| Mean images / identity | 6.74 |
| Real enrolments | 6 identities / 30 images |
| Benchmark (Olivetti) | 40 identities / 280 images |

## 2. Image Properties

| Metric | Value |
|---|---|
| Distinct resolutions | 2 |
| Mean resolution | 0.367 MP |
| Mean file size | 18.9 KB |
| Mean aspect ratio | 1.376 |
| Unreadable files | 0 |

## 3. Face Detection & Quality

| Metric | Value |
|---|---|
| Face-detection rate | 100.0%  (310/310) |
| Mean quality score | 0.446 |
| Mean sharpness (norm.) | 0.029 |
| Mean face-size ratio | 0.317 |
| Images below min_quality (0.15) | 0 |

## 4. Embedding Space (ArcFace 512-d)

| Metric | Value |
|---|---|
| Embeddings computed | 310 |
| Dimensionality | 512 |
| Mean L2 norm (expect 1.0) | 1.000000 |
| Max deviation from unit norm | 1.19e-07 |

## 5. Class Separability (genuine vs impostor cosine similarity, by domain)

The two image domains are analysed **separately** because they are not
comparable: Olivetti is grayscale 64×64 upscaled (a deliberate regression/stress
set), while the real enrolments are colour webcam captures — the true deployment
domain. `d′` is the standardised separation between the genuine and impostor
similarity distributions (higher = more separable).

| Domain | Genuine μ | Impostor μ | d′ | Impostor pairs ≥ 0.35 |
|---|---|---|---|---|
| Real webcam enrolments | 0.434 | 0.817 | -2.25 | 15 / 15 |
| Olivetti benchmark (grayscale 64px) | 0.716 | 0.775 | -0.59 | 780 / 780 |

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
| students | 3 |
| student_embeddings | 21 |
| image_jobs | 4 |
| attendance_events | 4 |
| esp32_devices | 0 |
| fairness_metrics | 0 |

Attendance status breakdown: `{'recognized': 4}`
Image-job status breakdown: `{'complete': 4}`

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
