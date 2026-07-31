"""Recognition routes — JSON with base64 images (mobile and web clients)."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from faceapi import services
from faceapi.config import Settings, get_settings
from faceapi.deps import b64_to_frame, get_engine, get_store, require_key
from faceapi.engine import FaceEngine
from faceapi.schemas import EmbedResponse, EnrollResponse, IdentifyResponse, VerifyResponse
from faceapi.store import SubjectStore

router = APIRouter(prefix="/v1", tags=["recognition (json)"],
                   dependencies=[Depends(require_key)])


class ImageBody(BaseModel):
    image: str                       # base64, optionally a data: URI


class VerifyBody(ImageBody):
    subject_id: str


class EnrollBody(BaseModel):
    subject_id: str
    name: str | None = None
    metadata: dict | None = None
    images: list[str]


@router.post("/enroll", response_model=EnrollResponse)
async def enroll_json(body: EnrollBody, engine: FaceEngine = Depends(get_engine),
                      store: SubjectStore = Depends(get_store),
                      settings: Settings = Depends(get_settings)):
    frames = [b64_to_frame(engine, b, settings) for b in body.images]
    return services.enroll_subject(engine, store, body.subject_id, body.name,
                                   body.metadata, frames, settings)


@router.post("/identify", response_model=IdentifyResponse)
async def identify_json(body: ImageBody, engine: FaceEngine = Depends(get_engine),
                        store: SubjectStore = Depends(get_store),
                        settings: Settings = Depends(get_settings)):
    return services.identify_face(engine, store, b64_to_frame(engine, body.image, settings),
                                  settings)


@router.post("/verify", response_model=VerifyResponse)
async def verify_json(body: VerifyBody, engine: FaceEngine = Depends(get_engine),
                      store: SubjectStore = Depends(get_store),
                      settings: Settings = Depends(get_settings)):
    return services.verify_face(engine, store, body.subject_id,
                                b64_to_frame(engine, body.image, settings), settings)


@router.post("/embed", response_model=EmbedResponse)
async def embed_json(body: ImageBody, engine: FaceEngine = Depends(get_engine),
                     settings: Settings = Depends(get_settings)):
    return services.embed_face(engine, b64_to_frame(engine, body.image, settings), settings)
