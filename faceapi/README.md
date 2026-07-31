# Face Recognition API

A production-grade, **domain-agnostic** face-recognition microservice — enrol
subjects, then verify (1:1) or identify (1:N). Not tied to attendance; the
attendance system is just one possible client.

Pipeline: **YuNet** detection (5-point landmarks) → **ArcFace** 512-d embedding →
multi-prototype cosine matching with a calibrated threshold. Built for
**CPU servers and edge SBCs** (Raspberry-Pi class).

## Features
- `POST /enroll` · `POST /identify` (1:N) · `POST /verify` (1:1) · `POST /embed`
- `GET /subjects` · `DELETE /subjects/{id}` · `GET /models` · `GET /health`
- **Multi-prototype** galleries (keeps every enrol shot, scores by max similarity)
- **Flip test-time augmentation** (opt-in, +accuracy)
- **Domain calibration** — pick the threshold at a target false-match-rate
- **Low-resource profile** — tiny 13.6 MB model by default; opt-in INT8 (~3.9× smaller)
- **API-key auth**, request validation, OpenAPI docs at `/docs`
- Pure-stdlib SQLite gallery (light on edge)

## Quick start
```bash
pip install -r requirements-faceapi.txt
python -m worker.download_models        # one-time: fetch YuNet + ArcFace ONNX
python -m faceapi.serve                 # http://localhost:8080  (docs at /docs)
```

Enrol and identify with curl:
```bash
curl -F subject_id=alice -F name=Alice \
     -F images=@a1.jpg -F images=@a2.jpg -F images=@a3.jpg \
     http://localhost:8080/enroll

curl -F image=@query.jpg http://localhost:8080/identify
```

## Configuration (env, prefix `FACEAPI_`)
| Var | Default | Purpose |
|---|---|---|
| `FACEAPI_MATCH_THRESHOLD` | `0.42` | cosine cut-off — **calibrate for your domain** |
| `FACEAPI_ENABLE_TTA` | `false` | flip test-time augmentation |
| `FACEAPI_USE_INT8` | `false` | use the quantized model (memory-limited edge) |
| `FACEAPI_INTRA_OP_THREADS` | `0` (auto) | pin threads (e.g. `4` on a Pi) |
| `FACEAPI_API_KEYS` | `` (off) | comma-separated keys; required in `X-API-Key` |
| `FACEAPI_DB_PATH` | `work/faceapi.db` | subject gallery |

## Optimize + calibrate for your data
```bash
# 1. calibrate the threshold on labelled folders (one subfolder per person)
python -m faceapi.calibrate --dir work/my_people --fmr 0.01
#    -> prints e.g.  FACEAPI_MATCH_THRESHOLD=0.55

# 2. (optional, edge) build the INT8 model — ~3.9x smaller
python -m faceapi.quantize     # then run with FACEAPI_USE_INT8=true
```
> Note: INT8 is a **memory** win; on x86 CPUs it can be *slower* (ORT INT8 kernels),
> while ARM SBCs often speed up — always benchmark on your target with `faceapi.quantize`.

## Low-resource notes
- The default `w600k_mbf` model is already tiny (**13.6 MB, ~5 ms/face** on desktop CPU)
  and is the recommended edge choice; avoid the 174 MB `w600k_r50` on constrained boards.
- Detection dominates latency and isn't batched — for more speed use a quantised YuNet or
  an accelerated execution provider (OpenVINO / GPU) where available.

## Docker
```bash
docker build -f faceapi/Dockerfile -t faceapi .
docker run -p 8080:8080 -e FACEAPI_API_KEYS=changeme faceapi
```

## Cross-platform — works everywhere
The service is a plain HTTP API with **CORS enabled** and two interchangeable input
styles, so it drives the same from any client:

- **multipart/form-data** — browser `<form>`, `FormData`, Flutter `MultipartRequest`
- **application/json + base64** (`/v1/*`) — easiest for mobile/web/desktop

A ready-made **live demo** (camera capture + enrol + identify, responsive for
**phone / tablet / desktop**) is served at **`GET /`**; interactive API docs at `/docs`.

**Web / JavaScript**
```js
const b64 = canvas.toDataURL("image/jpeg");            // data: URI ok
const r = await fetch("/v1/identify", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ image: b64 })
});
const { status, match } = await r.json();
```

**Flutter / Dart (mobile)**
```dart
final b64 = base64Encode(await File(path).readAsBytes());
final r = await http.post(Uri.parse("$api/v1/identify"),
  headers: {"Content-Type": "application/json", "X-API-Key": key},
  body: jsonEncode({"image": b64}));
```

**Swift / iOS**
```swift
let b64 = imageData.base64EncodedString()
var req = URLRequest(url: URL(string: "\(api)/v1/identify")!)
req.httpMethod = "POST"; req.setValue("application/json", forHTTPHeaderField: "Content-Type")
req.httpBody = try JSONSerialization.data(withJSONObject: ["image": b64])
```

**curl / desktop / CI**
```bash
curl -F image=@query.jpg http://localhost:8080/identify           # multipart
curl -H 'Content-Type: application/json' \
     -d "{\"image\":\"$(base64 -w0 query.jpg)\"}" \
     http://localhost:8080/v1/identify                            # json
```

Generate typed client SDKs for any language from the OpenAPI schema at
`/openapi.json` (e.g. `openapi-generator` for Kotlin/Swift/TypeScript/Dart).

## Observability
- Every response carries `X-Request-ID` and `X-Process-Time-ms`.
- `GET /metrics` exposes Prometheus counters (`faceapi_requests_total`,
  `faceapi_errors_total`, `faceapi_request_duration_seconds_sum`).
- Consistent error envelope: `{"error": {"code", "message", "path"}}`.

See `MODEL_CARD.md` for performance, limitations, and ethical/privacy guidance.

## Tests
```bash
python -m pytest faceapi/tests -q      # in-process API coverage (multipart + json + metrics)
python -m faceapi.smoke                # end-to-end demo on Olivetti faces
```
