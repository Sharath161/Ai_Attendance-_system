"""Production-grade, domain-agnostic Face Recognition API.

Endpoints
    POST /enroll          multipart: subject_id, name?, metadata?, images[]  -> EnrollResponse
    POST /identify        multipart: image                                    -> IdentifyResponse (1:N)
    POST /verify          multipart: subject_id, image                        -> VerifyResponse   (1:1)
    POST /embed           multipart: image                                    -> EmbedResponse
    GET  /subjects        list enrolled subjects
    DELETE /subjects/{id} remove a subject
    GET  /models          active model + runtime profile
    GET  /health          liveness + gallery stats

Auth: if FACEAPI_API_KEYS is set (comma-separated), all non-health routes
require a matching `X-API-Key` header.
"""
from __future__ import annotations

import json
from typing import Annotated

import numpy as np
from fastapi import (Depends, FastAPI, File, Form, Header, HTTPException,
                     UploadFile, status)

from faceapi import matching
from faceapi.config import Settings, get_settings
from faceapi.engine import FaceEngine
from faceapi.schemas import (BBox, Candidate, EmbedResponse, EnrollResponse,
                             HealthResponse, IdentifyResponse, SubjectInfo,
                             VerifyResponse)
from faceapi.store import SubjectStore

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    app.state.engine = FaceEngine(
        detection_model=s.detection_model_path,
        recognition_model=s.recognition_model(),
        detection_score_threshold=s.detection_score_threshold,
        intra_op_threads=s.intra_op_threads,
        enable_tta=s.enable_tta)
    app.state.store = SubjectStore(s.db_path)
    yield
    try:
        app.state.store.close()
    except Exception:
        pass


app = FastAPI(title="Face Recognition API", version="1.0", lifespan=lifespan,
              description="Domain-agnostic face enrolment / verification / identification.")


# ── auth ──────────────────────────────────────────────────────────────────────
def require_key(x_api_key: Annotated[str | None, Header()] = None,
                settings: Settings = Depends(get_settings)) -> None:
    keys = settings.api_key_set()
    if keys and x_api_key not in keys:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key")


def engine() -> FaceEngine: return app.state.engine
def store() -> SubjectStore: return app.state.store


async def _read(image: UploadFile, settings: Settings) -> np.ndarray:
    data = await image.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "Image too large")
    frame = app.state.engine.decode(data)
    if frame is None:
        raise HTTPException(400, "Could not decode image")
    return frame


# ── enrol ─────────────────────────────────────────────────────────────────────
@app.post("/enroll", response_model=EnrollResponse, dependencies=[Depends(require_key)])
async def enroll(
    subject_id: str = Form(...),
    name: str | None = Form(None),
    metadata: str | None = Form(None),
    images: list[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
) -> EnrollResponse:
    meta = None
    if metadata:
        try: meta = json.loads(metadata)
        except Exception: raise HTTPException(400, "metadata must be JSON")

    vectors, rejected = [], 0
    for img in images:
        frame = await _read(img, settings)
        dets = engine().detect(frame, embed=True, max_faces=settings.max_faces)
        if len(dets) == 1 and dets[0].embedding is not None:
            vectors.append(dets[0].embedding)
        else:
            rejected += 1
    if not vectors:
        raise HTTPException(422, "No usable single-face images (each photo needs exactly one face)")

    store().upsert_subject(subject_id, name, meta)
    added = store().add_embeddings(subject_id, vectors, engine().recognition_model_name)
    existing = store().subject_vectors(subject_id)
    total = 0 if existing is None else len(existing)
    return EnrollResponse(subject_id=subject_id, name=name, embeddings_added=added,
                          total_embeddings=total, faces_rejected=rejected)


# ── identify (1:N) ────────────────────────────────────────────────────────────
@app.post("/identify", response_model=IdentifyResponse, dependencies=[Depends(require_key)])
async def identify(image: UploadFile = File(...), settings: Settings = Depends(get_settings)):
    frame = await _read(image, settings)
    dets = engine().detect(frame, embed=True, max_faces=settings.max_faces)
    thr = settings.match_threshold
    if not dets:
        return IdentifyResponse(status="no_face", threshold=thr)
    if len(dets) > 1:
        return IdentifyResponse(status="multiple_faces", face_box=BBox(**dets[0].bbox), threshold=thr)
    res = matching.identify(dets[0].embedding, store().gallery(), thr)
    cands = [Candidate(**c) for c in res["candidates"]]
    match = Candidate(**res["match"]) if res["match"] else None
    return IdentifyResponse(status="matched" if match else "no_match", match=match,
                            candidates=cands, face_box=BBox(**dets[0].bbox), threshold=thr)


# ── verify (1:1) ──────────────────────────────────────────────────────────────
@app.post("/verify", response_model=VerifyResponse, dependencies=[Depends(require_key)])
async def verify(subject_id: str = Form(...), image: UploadFile = File(...),
                 settings: Settings = Depends(get_settings)):
    frame = await _read(image, settings)
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


# ── embed ─────────────────────────────────────────────────────────────────────
@app.post("/embed", response_model=EmbedResponse, dependencies=[Depends(require_key)])
async def embed(image: UploadFile = File(...), settings: Settings = Depends(get_settings)):
    frame = await _read(image, settings)
    dets = engine().detect(frame, embed=True, max_faces=settings.max_faces)
    if not dets:
        return EmbedResponse(face_count=0, embedding=None, model=engine().recognition_model_name, dim=512)
    d = dets[0]
    return EmbedResponse(face_count=len(dets), embedding=[round(float(x), 6) for x in d.embedding],
                         face_box=BBox(**d.bbox), model=engine().recognition_model_name, dim=512)


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
