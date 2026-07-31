"""Shared FastAPI dependencies and small helpers used across routers."""
from __future__ import annotations

import base64

from fastapi import HTTPException, Request, UploadFile

from attendance.config import Settings
from attendance.db import Database
from attendance.faceclient import FaceClient


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_face(request: Request) -> FaceClient:
    return request.app.state.face


def user_out(row) -> dict:
    """SQLite row -> public user dict (never leaks password_hash)."""
    return {
        "id": row["id"],
        "email": row["email"],
        "full_name": row["full_name"],
        "role": row["role"],
        "student_number": row["student_number"],
        "face_enrolled": bool(row["face_enrolled"]),
    }


def b64_to_bytes(b64: str) -> bytes:
    """Decode a base64 payload, tolerating a data: URI prefix."""
    if "," in b64[:64]:
        b64 = b64.split(",", 1)[1]
    try:
        return base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 image")


async def read_uploads(images: list[UploadFile], settings: Settings) -> list[bytes]:
    out: list[bytes] = []
    for img in images:
        data = await img.read()
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, f"{img.filename} exceeds the maximum upload size")
        out.append(data)
    return out
