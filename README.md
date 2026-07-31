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

| URL | What |
|---|---|
| **http://localhost:8080** | Product site — overview, benchmarks, code examples |
| **/demo** | Live camera demo (enrol + identify in the browser) |
| **/docs** | Interactive API reference (OpenAPI) |

```bash
curl -F subject_id=alice -F name=Alice -F images=@a1.jpg -F images=@a2.jpg \
     http://localhost:8080/enroll
curl -F image=@query.jpg http://localhost:8080/identify
```

## Built with it

The **Smart Attendance** system is a complete product running on this API — students
enrol and check in by face, staff manage sessions and reports. It owns zero ML code;
every face operation is an HTTP call to this service.

```bash
python -m attendance.serve     # http://localhost:8000
```

## Project structure

Backend and frontend are separated, and each backend is layered by responsibility.

```
faceapi/                     Recognition service (the product)
├── engine.py                detect · align · embed        (no HTTP)
├── store.py  matching.py    gallery persistence · scoring
├── services.py              recognition operations
├── deps.py  middleware.py   DI helpers · observability
├── routers/                 HTTP surface
│   ├── recognition.py       multipart endpoints
│   ├── recognition_json.py  JSON/base64 endpoints (/v1)
│   ├── subjects.py          gallery management
│   └── system.py            health · models · metrics
├── api.py                   app factory (wiring only)
└── web/                     frontend
    ├── index.html demo.html
    ├── css/                 site.css · demo.css
    └── js/                  site.js · demo.js

attendance/                  Application built on the API
├── db.py                    persistence
├── auth.py                  JWT · Argon2 · role guards
├── faceclient.py            HTTP client for faceapi
├── schemas.py  deps.py      request models · DI helpers
├── services/checkin.py      check-in business rules
├── routers/                 auth · face · courses · sessions
│                            attendance · reports · system
├── app.py                   app factory (wiring only)
└── web/                     PWA frontend
    ├── index.html
    ├── css/styles.css
    └── js/
        ├── main.js          shell · routing · boot
        ├── api.js           HTTP client + token store
        ├── ui.js            DOM helpers
        ├── camera.js        capture modal
        └── views/           student.js · staff.js · admin.js
```

## Documentation
- [`faceapi/README.md`](faceapi/README.md) — endpoints, configuration, cross-platform client snippets, Docker
- [`faceapi/MODEL_CARD.md`](faceapi/MODEL_CARD.md) — intended use, measured metrics, limitations, bias & privacy

## Tests
```bash
python -m pytest faceapi/tests -q     # API coverage
python -m faceapi.smoke               # end-to-end demo
```
