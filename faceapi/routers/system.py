"""Health, model info and metrics routes (public — no API key required)."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from faceapi.config import Settings, get_settings
from faceapi.deps import get_engine, get_store
from faceapi.engine import FaceEngine
from faceapi.middleware import prometheus_text
from faceapi.schemas import HealthResponse
from faceapi.store import SubjectStore

router = APIRouter(tags=["system"])


@router.get("/models")
async def models(engine: FaceEngine = Depends(get_engine),
                 settings: Settings = Depends(get_settings)):
    return {"recognition_model": engine.recognition_model_name, "dim": 512,
            "int8": settings.use_int8, "tta": settings.enable_tta,
            "threshold": settings.match_threshold,
            "intra_op_threads": settings.intra_op_threads or "auto"}


@router.get("/health", response_model=HealthResponse)
async def health(engine: FaceEngine = Depends(get_engine),
                 store: SubjectStore = Depends(get_store),
                 settings: Settings = Depends(get_settings)):
    subjects, embeddings = store.count()
    return HealthResponse(status="ok", model=engine.recognition_model_name,
                          int8=settings.use_int8, tta=settings.enable_tta,
                          threshold=settings.match_threshold,
                          subjects=subjects, embeddings=embeddings)


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return prometheus_text()
