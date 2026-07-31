# Face Recognition API

A production-grade, **domain-agnostic** face recognition service built for
**low-data enrolment** and **low-hardware deployment**.

Enrol a person from **as little as one photo**, then verify (1:1) or identify
(1:N) from any client — web, mobile, desktop.

```
YuNet detection (5-pt landmarks) → ArcFace 512-d embedding → multi-prototype cosine match
```

## Measured performance

LFW, 120 enrolled identities, fixed 5-photo test set per identity, 114 unknown probes:

| Enrolment photos | Top-1 accuracy | Open-set DIR @ FAR=1% |
|---|---|---|
| **1** | **97.5%** | 97.0% |
| **2** | **98.5%** | 97.7% |
| 5 | 98.5% | **98.2%** |

~9.6 ms/image end-to-end on desktop CPU · 13.6 MB recognition model ·
sub-10 ms matching at 200k enrolled prototypes.

See [`faceapi/benchmarks/results/FEWSHOT_REPORT.md`](faceapi/benchmarks/results/FEWSHOT_REPORT.md)
and [`faceapi/MODEL_CARD.md`](faceapi/MODEL_CARD.md) for the full protocol,
limitations and ethical guidance.

## Quick start

```bash
pip install -r requirements.txt
python -m faceapi.download_models     # one-time (~14 MB)
python -m faceapi.serve               # http://localhost:8080
```

Open **http://localhost:8080** for the live camera demo, or **/docs** for the
interactive API reference.

```bash
curl -F subject_id=alice -F name=Alice -F images=@a1.jpg -F images=@a2.jpg \
     http://localhost:8080/enroll
curl -F image=@query.jpg http://localhost:8080/identify
```

## Documentation
- [`faceapi/README.md`](faceapi/README.md) — endpoints, configuration, cross-platform client snippets, Docker
- [`faceapi/MODEL_CARD.md`](faceapi/MODEL_CARD.md) — intended use, measured metrics, limitations, bias & privacy

## Tests
```bash
python -m pytest faceapi/tests -q     # API coverage
python -m faceapi.smoke               # end-to-end demo
```
