"""Face enrolment routes — delegate to the recognition service."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from attendance.auth import current_user, require_staff
from attendance.config import Settings, get_settings
from attendance.db import Database
from attendance.deps import b64_to_bytes, get_db, get_face, read_uploads, user_out
from attendance.faceclient import FaceClient, FaceServiceError
from attendance.schemas import EnrolFaceBody

router = APIRouter(prefix="/face", tags=["face"])


@router.post("/enroll")
async def enroll_own_face(body: EnrolFaceBody, db: Database = Depends(get_db),
                          face: FaceClient = Depends(get_face), user=Depends(current_user)):
    """Enrol the caller's own face (JSON/base64 — used by the PWA camera)."""
    if not body.images:
        raise HTTPException(400, "At least one image is required")
    try:
        res = await face.enroll(user["face_subject_id"], user["full_name"],
                                [b64_to_bytes(b) for b in body.images])
    except FaceServiceError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc))
    db.mark_face_enrolled(user["id"], True)
    return {"enrolled": True, "embeddings_added": res.get("embeddings_added"),
            "faces_rejected": res.get("faces_rejected")}


@router.post("/enroll/{user_id}")
async def enroll_for_user(user_id: str, images: list[UploadFile] = File(...),
                          db: Database = Depends(get_db),
                          face: FaceClient = Depends(get_face),
                          settings: Settings = Depends(get_settings),
                          _=Depends(require_staff)):
    """Staff enrol a student from uploaded photo files (multipart)."""
    target = db.user_by_id(user_id)
    if target is None:
        raise HTTPException(404, "User not found")
    data = await read_uploads(images, settings)
    try:
        res = await face.enroll(target["face_subject_id"], target["full_name"], data)
    except FaceServiceError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc))
    db.mark_face_enrolled(user_id, True)
    return {"enrolled": True, "user": user_out(db.user_by_id(user_id)),
            "embeddings_added": res.get("embeddings_added")}
