"""Shared dependencies and decoding helpers for the recognition API."""
from __future__ import annotations

import base64
import binascii

import numpy as np
from fastapi import Depends, Header, HTTPException, Request, status
from typing import Annotated

from faceapi.config import Settings, get_settings
from faceapi.engine import FaceEngine
from faceapi.store import SubjectStore


def get_engine(request: Request) -> FaceEngine:
    return request.app.state.engine


def get_store(request: Request) -> SubjectStore:
    return request.app.state.store


def require_key(x_api_key: Annotated[str | None, Header()] = None,
                settings: Settings = Depends(get_settings)) -> None:
    """Enforce X-API-Key when FACEAPI_API_KEYS is configured."""
    keys = settings.api_key_set()
    if keys and x_api_key not in keys:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key")


def decode_frame(engine: FaceEngine, data: bytes, settings: Settings) -> np.ndarray:
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "Image too large")
    frame = engine.decode(data)
    if frame is None:
        raise HTTPException(400, "Could not decode image")
    return frame


def b64_to_frame(engine: FaceEngine, b64: str, settings: Settings) -> np.ndarray:
    if "," in b64[:64]:                       # strip data: URI prefix
        b64 = b64.split(",", 1)[1]
    try:
        return decode_frame(engine, base64.b64decode(b64), settings)
    except (binascii.Error, ValueError):
        raise HTTPException(400, "Invalid base64 image")
