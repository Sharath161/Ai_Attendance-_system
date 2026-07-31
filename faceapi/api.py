"""Production-grade, cross-platform Face Recognition API.

Callable from anywhere — web (fetch/XHR), mobile (Flutter/Swift/Kotlin),
desktop, curl — via two interchangeable input styles:

  * multipart/form-data  (browser <form>, FormData, MultipartRequest)
  * application/json      with base64-encoded images  (easy for mobile/web)

Endpoints
    POST /enroll                subject_id,name?,metadata?,images[]        (multipart)
    POST /v1/enroll             {subject_id,name?,metadata?,images:[b64]}  (json)
    POST /identify | /v1/identify        image / {image:b64}      -> 1:N
    POST /verify   | /v1/verify          subject_id,image / json  -> 1:1
    POST /embed    | /v1/embed           image / {image:b64}      -> 512-d
    GET  /subjects · DELETE /subjects/{id} · GET /models · GET /health
    GET  /metrics   (Prometheus text)   ·   GET /   (live web demo)

Auth: if FACEAPI_API_KEYS is set, non-public routes need a matching X-API-Key.
CORS is open by default (configure FACEAPI_CORS_ORIGINS in production).
"""
from __future__ import annotations

import base64
import binascii
import json
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import (Depends, FastAPI, File, Form, Header, HTTPException,
                     Request, UploadFile, status)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

from faceapi import matching
from faceapi.config import Settings, get_settings
from faceapi.engine import FaceEngine
from faceapi.schemas import (BBox, Candidate, EmbedResponse, EnrollResponse,
                             HealthResponse, IdentifyResponse, SubjectInfo,
                             VerifyResponse)
from faceapi.store import SubjectStore

_WEB = Path(__file__).resolve().parent / "web"

# ── lightweight metrics ────────────────────────────────────────────────────────
_METRICS = {"requests": defaultdict(int), "errors": defaultdict(int),
            "latency_sum": defaultdict(float)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    app.state.engine = FaceEngine(
        detection_model=s.detection_model_path, recognition_model=s.recognition_model(),
        detection_score_threshold=s.detection_score_threshold,
        intra_op_threads=s.intra_op_threads, enable_tta=s.enable_tta,
        detect_max_side=s.detect_max_side)
    app.state.store = SubjectStore(s.db_path)
    yield
    try:
        app.state.store.close()
    except Exception:
        pass


app = FastAPI(title="Face Recognition API", version="1.0", lifespan=lifespan,
              description="Cross-platform face enrolment / verification / identification.")

_settings = get_settings()
_cors = _settings.cors_origins.split(",") if _settings.cors_origins else ["*"]
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in _cors],
                   allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def observability(request: Request, call_next):
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    t0 = time.perf_counter()
    try:
        resp = await call_next(request)
    except Exception:
        _METRICS["errors"][request.url.path] += 1
        raise
    dt = time.perf_counter() - t0
    ep = request.url.path
    _METRICS["requests"][ep] += 1
    _METRICS["latency_sum"][ep] += dt
    if resp.status_code >= 400:
        _METRICS["errors"][ep] += 1
    resp.headers["X-Request-ID"] = rid
    resp.headers["X-Process-Time-ms"] = f"{dt*1000:.1f}"
    return resp


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code,
                        content={"error": {"code": exc.status_code, "message": exc.detail,
                                           "path": request.url.path}})


# ── auth + accessors ───────────────────────────────────────────────────────────
def require_key(x_api_key: Annotated[str | None, Header()] = None,
                settings: Settings = Depends(get_settings)) -> None:
    keys = settings.api_key_set()
    if keys and x_api_key not in keys:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key")


def engine() -> FaceEngine: return app.state.engine
def store() -> SubjectStore: return app.state.store


def _decode_frame(data: bytes, settings: Settings) -> np.ndarray:
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "Image too large")
    frame = engine().decode(data)
    if frame is None:
        raise HTTPException(400, "Could not decode image")
    return frame


def _b64_to_frame(b64: str, settings: Settings) -> np.ndarray:
    if "," in b64[:64]:                      # strip data: URI prefix
        b64 = b64.split(",", 1)[1]
    try:
        return _decode_frame(base64.b64decode(b64), settings)
    except (binascii.Error, ValueError):
        raise HTTPException(400, "Invalid base64 image")


# ── core logic (shared by multipart + json routes) ────────────────────────────
def _do_embed(frame, settings) -> EmbedResponse:
    dets = engine().detect(frame, embed=True, max_faces=settings.max_faces)
    if not dets:
        return EmbedResponse(face_count=0, embedding=None, model=engine().recognition_model_name, dim=512)
    d = dets[0]
    return EmbedResponse(face_count=len(dets), embedding=[round(float(x), 6) for x in d.embedding],
                         face_box=BBox(**d.bbox), model=engine().recognition_model_name, dim=512)


def _do_identify(frame, settings) -> IdentifyResponse:
    dets = engine().detect(frame, embed=True, max_faces=settings.max_faces)
    thr = settings.match_threshold
    if not dets:
        return IdentifyResponse(status="no_face", threshold=thr)
    if len(dets) > 1:
        return IdentifyResponse(status="multiple_faces", face_box=BBox(**dets[0].bbox), threshold=thr)
    res = matching.identify(dets[0].embedding, store().gallery(), thr)
    match = Candidate(**res["match"]) if res["match"] else None
    return IdentifyResponse(status="matched" if match else "no_match", match=match,
                            candidates=[Candidate(**c) for c in res["candidates"]],
                            face_box=BBox(**dets[0].bbox), threshold=thr)


def _do_verify(subject_id, frame, settings) -> VerifyResponse:
    thr = settings.match_threshold
    mat = store().subject_vectors(subject_id)
    if mat is None:
        return VerifyResponse(status="unknown_subject", threshold=thr)
    dets = engine().detect(frame, embed=True, max_faces=settings.max_faces)
    if not dets:
        return VerifyResponse(status="no_face", threshold=thr)
    if len(dets) > 1:
        return VerifyResponse(status="multiple_faces", threshold=thr)
    r = matching.verify(dets[0].embedding, mat, thr)
    return VerifyResponse(status="verified" if r["is_match"] else "rejected",
                          is_match=r["is_match"], score=round(r["score"], 4), threshold=thr)


def _do_enroll(subject_id, name, meta, frames, settings) -> EnrollResponse:
    from faceapi.fewshot import amplified_embeddings
    vectors, rejected = [], 0
    for frame in frames:
        crops = engine().align_crops(frame, max_faces=settings.max_faces)
        if len(crops) != 1:
            rejected += 1
            continue
        if settings.enroll_amplify:
            # few-shot amplifier: 1 photo -> ~6 augmented prototype embeddings
            vectors.extend(amplified_embeddings(engine(), crops[0]["crop"], settings.embed_batch))
        else:
            vectors.append(engine().embed_crops([crops[0]["crop"]])[0])
    if not vectors:
        raise HTTPException(422, "No usable single-face images (each photo needs exactly one face)")
    store().upsert_subject(subject_id, name, meta)
    added = store().add_embeddings(subject_id, vectors, engine().recognition_model_name)
    existing = store().subject_vectors(subject_id)
    return EnrollResponse(subject_id=subject_id, name=name, embeddings_added=added,
                          total_embeddings=0 if existing is None else len(existing),
                          faces_rejected=rejected)


# ── JSON request bodies ────────────────────────────────────────────────────────
class ImageBody(BaseModel):
    image: str                       # base64 (optionally data: URI)


class VerifyBody(ImageBody):
    subject_id: str


class EnrollBody(BaseModel):
    subject_id: str
    name: str | None = None
    metadata: dict | None = None
    images: list[str]                # base64 list


# ── multipart routes ───────────────────────────────────────────────────────────
@app.post("/enroll", response_model=EnrollResponse, dependencies=[Depends(require_key)])
async def enroll(subject_id: str = Form(...), name: str | None = Form(None),
                 metadata: str | None = Form(None), images: list[UploadFile] = File(...),
                 settings: Settings = Depends(get_settings)):
    meta = None
    if metadata:
        try: meta = json.loads(metadata)
        except Exception: raise HTTPException(400, "metadata must be JSON")
    frames = [_decode_frame(await i.read(), settings) for i in images]
    return _do_enroll(subject_id, name, meta, frames, settings)


@app.post("/identify", response_model=IdentifyResponse, dependencies=[Depends(require_key)])
async def identify(image: UploadFile = File(...), settings: Settings = Depends(get_settings)):
    return _do_identify(_decode_frame(await image.read(), settings), settings)


@app.post("/verify", response_model=VerifyResponse, dependencies=[Depends(require_key)])
async def verify(subject_id: str = Form(...), image: UploadFile = File(...),
                 settings: Settings = Depends(get_settings)):
    return _do_verify(subject_id, _decode_frame(await image.read(), settings), settings)


@app.post("/embed", response_model=EmbedResponse, dependencies=[Depends(require_key)])
async def embed(image: UploadFile = File(...), settings: Settings = Depends(get_settings)):
    return _do_embed(_decode_frame(await image.read(), settings), settings)


# ── JSON (base64) routes — mobile/web friendly ────────────────────────────────
@app.post("/v1/enroll", response_model=EnrollResponse, dependencies=[Depends(require_key)])
async def enroll_json(body: EnrollBody, settings: Settings = Depends(get_settings)):
    frames = [_b64_to_frame(b, settings) for b in body.images]
    return _do_enroll(body.subject_id, body.name, body.metadata, frames, settings)


@app.post("/v1/identify", response_model=IdentifyResponse, dependencies=[Depends(require_key)])
async def identify_json(body: ImageBody, settings: Settings = Depends(get_settings)):
    return _do_identify(_b64_to_frame(body.image, settings), settings)


@app.post("/v1/verify", response_model=VerifyResponse, dependencies=[Depends(require_key)])
async def verify_json(body: VerifyBody, settings: Settings = Depends(get_settings)):
    return _do_verify(body.subject_id, _b64_to_frame(body.image, settings), settings)


@app.post("/v1/embed", response_model=EmbedResponse, dependencies=[Depends(require_key)])
async def embed_json(body: ImageBody, settings: Settings = Depends(get_settings)):
    return _do_embed(_b64_to_frame(body.image, settings), settings)


# ── management ────────────────────────────────────────────────────────────────
@app.get("/subjects", response_model=list[SubjectInfo], dependencies=[Depends(require_key)])
async def subjects():
    return [SubjectInfo(**s) for s in store().list_subjects()]


@app.delete("/subjects/{subject_id}", dependencies=[Depends(require_key)])
async def delete_subject(subject_id: str):
    if not store().delete_subject(subject_id):
        raise HTTPException(404, "Subject not found")
    return {"deleted": subject_id}


@app.get("/models")
async def models(settings: Settings = Depends(get_settings)):
    return {"recognition_model": engine().recognition_model_name, "dim": 512,
            "int8": settings.use_int8, "tta": settings.enable_tta,
            "threshold": settings.match_threshold,
            "intra_op_threads": settings.intra_op_threads or "auto"}


@app.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings)):
    subs, embs = store().count()
    return HealthResponse(status="ok", model=engine().recognition_model_name,
                          int8=settings.use_int8, tta=settings.enable_tta,
                          threshold=settings.match_threshold, subjects=subs, embeddings=embs)


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    lines = ["# HELP faceapi_requests_total Requests per endpoint",
             "# TYPE faceapi_requests_total counter"]
    for ep, n in _METRICS["requests"].items():
        lines.append(f'faceapi_requests_total{{endpoint="{ep}"}} {n}')
    for ep, n in _METRICS["errors"].items():
        lines.append(f'faceapi_errors_total{{endpoint="{ep}"}} {n}')
    for ep, s in _METRICS["latency_sum"].items():
        lines.append(f'faceapi_request_duration_seconds_sum{{endpoint="{ep}"}} {s:.4f}')
    return "\n".join(lines) + "\n"


@app.get("/", include_in_schema=False)
async def product_site():
    """Marketing / product site for the API."""
    page = _WEB / "index.html"
    if page.exists():
        return FileResponse(page, media_type="text/html")
    return JSONResponse({"service": "Face Recognition API", "docs": "/docs"})


@app.get("/demo", include_in_schema=False)
async def web_demo():
    """Interactive camera demo (enrol + identify from the browser)."""
    page = _WEB / "demo.html"
    if page.exists():
        return FileResponse(page, media_type="text/html")
    raise HTTPException(404, "Demo page not found")


@app.get("/site.css", include_in_schema=False)
async def site_css():
    return FileResponse(_WEB / "site.css", media_type="text/css")


@app.get("/icon.svg", include_in_schema=False)
async def site_icon():
    icon = _WEB / "icon.svg"
    if icon.exists():
        return FileResponse(icon, media_type="image/svg+xml")
    raise HTTPException(404, "icon not found")
