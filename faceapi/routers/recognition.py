"""Recognition routes — multipart/form-data (browsers, curl, file uploads)."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from faceapi import services
from faceapi.config import Settings, get_settings
from faceapi.deps import decode_frame, get_engine, get_store, require_key
from faceapi.engine import FaceEngine
from faceapi.schemas import EmbedResponse, EnrollResponse, IdentifyResponse, VerifyResponse
from faceapi.store import SubjectStore

router = APIRouter(tags=["recognition"], dependencies=[Depends(require_key)])


@router.post("/enroll", response_model=EnrollResponse)
async def enroll(subject_id: str = Form(...), name: str | None = Form(None),
                 metadata: str | None = Form(None), images: list[UploadFile] = File(...),
                 engine: FaceEngine = Depends(get_engine),
                 store: SubjectStore = Depends(get_store),
                 settings: Settings = Depends(get_settings)):
    meta = None
    if metadata:
        try:
            meta = json.loads(metadata)
        except Exception:
            raise HTTPException(400, "metadata must be JSON")
    frames = [decode_frame(engine, await i.read(), settings) for i in images]
    return services.enroll_subject(engine, store, subject_id, name, meta, frames, settings)


@router.post("/identify", response_model=IdentifyResponse)
async def identify(image: UploadFile = File(...),
                   engine: FaceEngine = Depends(get_engine),
                   store: SubjectStore = Depends(get_store),
                   settings: Settings = Depends(get_settings)):
    frame = decode_frame(engine, await image.read(), settings)
    return services.identify_face(engine, store, frame, settings)


@router.post("/verify", response_model=VerifyResponse)
async def verify(subject_id: str = Form(...), image: UploadFile = File(...),
                 engine: FaceEngine = Depends(get_engine),
                 store: SubjectStore = Depends(get_store),
                 settings: Settings = Depends(get_settings)):
    frame = decode_frame(engine, await image.read(), settings)
    return services.verify_face(engine, store, subject_id, frame, settings)


@router.post("/embed", response_model=EmbedResponse)
async def embed(image: UploadFile = File(...),
                engine: FaceEngine = Depends(get_engine),
                settings: Settings = Depends(get_settings)):
    frame = decode_frame(engine, await image.read(), settings)
    return services.embed_face(engine, frame, settings)
