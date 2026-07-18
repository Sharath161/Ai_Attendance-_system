# Performance & Scalability Benchmark — Smart Attendance System

**Project:** Develop an Efficient and Scalable Backend (Smart Attendance)
**Module:** MA981-7-FY · MSc Data Science and Its Applications · University of Essex
**Generated:** 2026-07-18 15:30 UTC
**Host:** 12 logical CPUs · Python 3.14.5 · onnxruntime CPUExecutionProvider
**Pipeline:** YuNet detect → ArcFace MobileNet (512-d) → cosine match

---

## 1. Inference Latency (per image, CPU)

| Stage | Mean (ms) |
|---|---|
| YuNet detection | 24.4 |
| ArcFace align + embed | 4.7 |
| **End-to-end** | **29.1** |
| End-to-end p50 / p95 / p99 | 20.4 / 59.7 / 70.9 |

Samples: 360 timed inferences.

## 2. Recognition Throughput (single CPU worker)

| Metric | Value |
|---|---|
| Sustained throughput | **29.2 images/sec** |
| Per hour | 105,087 images |
| Capacity per 2-hour batch window | 210,174 images |

The README specifies the batch worker runs on a ~2-hour cycle; a single CPU
worker clears **210,174 images** per cycle, so a cohort of a
few hundred students with several kiosk captures each is handled comfortably by
one worker, and the design scales horizontally by adding workers.

## 3. Matching Scalability (cosine similarity over enrolled gallery)

Because embeddings are L2-normalised, matching is a single dense matrix–vector
product (`gallery @ query`) handled by NumPy's BLAS — O(N·d) and cache-friendly.

| Gallery size | Mean (ms) | p95 (ms) | Throughput (q/s) |
|---|---|---|---|
| 100 | 0.005 | 0.006 | 195,008 |
| 1,000 | 0.206 | 0.659 | 4,862 |
| 10,000 | 0.668 | 1.156 | 1,498 |
| 50,000 | 2.341 | 2.889 | 427 |
| 100,000 | 4.612 | 5.560 | 217 |
| 200,000 | 8.759 | 9.945 | 114 |

Even at **200,000** enrolled prototypes a match completes in
~8.8 ms, so identity search is never the bottleneck; the
detector+embedder dominates end-to-end cost.

## 4. Optimizer Stage Timing

| Stage | Cost |
|---|---|
| Stage 1 — quality scoring | 24.6 ms / image |
| Stage 3 — LOO 80-threshold sweep (100 students) | 0.80 ms |

The full leave-one-out calibration is dominated by embedding extraction, not the
threshold sweep — the sweep itself is sub-millisecond even for hundreds of
students, so per-demographic re-calibration is cheap to re-run whenever the
cohort changes.

## Figures

1. `perf_fig1_latency_breakdown.png` — per-stage & end-to-end latency
2. `perf_fig2_throughput.png` — single-worker throughput
3. `perf_fig3_matching_scalability.png` — match latency/throughput vs gallery size
4. `perf_fig4_optimizer_stages.png` — optimizer stage timing
