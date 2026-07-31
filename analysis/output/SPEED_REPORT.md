# Speed Enhancement — recognition inference (CPU, 12 cores)

Baseline = the system's current default (ONNXRuntime multi-threaded) session.

| Config | ms/face | faces/sec | vs default |
|---|---|---|---|
| single, 1 thread (threads off, reference) | 16.67 | 60 | 0.35x |
| single, 12 threads (**current default**) | 5.79 | 173 | 1.00x |
| batch=8, 12 threads | 4.28 | 234 | 1.35x |
| batch=16, 12 threads | 4.52 | 221 | 1.28x |
| batch=32, 12 threads | 5.30 | 189 | 1.09x |
| batch=64, 12 threads | 5.07 | 197 | 1.14x |

YuNet **detection ≈ 15.8 ms/img** dominates end-to-end cost and is not batched through cv2.FaceDetectorYN, so end-to-end goes from ~22 ms/img to ~20 ms/img.

**Honest takeaways.**
- The pipeline is *already* multi-threaded, so threading is not a new win — it is the baseline.
- Batching the embedder adds only **1.4x** on the recognition step (best at batch=8), and because detection dominates, end-to-end improves ~1.5x at most.
- The genuine bottleneck is **detection** (~16 ms). The real speed levers are a quantised/INT8 YuNet or an accelerated execution provider (OpenVINO / GPU / DirectML) — none available on this CPU-only host.
- Practically the system is already comfortably within the 2-hour batch budget (~46 img/s/worker), so speed is not the binding constraint; batching mainly trims total worker runtime and cost.

Figure: `speed_batching.png`.