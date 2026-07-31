"""Health and service-status routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from attendance.deps import get_face
from attendance.faceclient import FaceClient

router = APIRouter(tags=["system"])


@router.get("/health")
async def health(face: FaceClient = Depends(get_face)):
    return {"status": "ok", "face_service": await face.health(),
            "face_service_url": face.base}
