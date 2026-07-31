"""Check-in and attendance-marking routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from attendance.auth import current_user, require_staff
from attendance.config import Settings, get_settings
from attendance.db import Database
from attendance.deps import b64_to_bytes, get_db, get_face
from attendance.faceclient import FaceClient
from attendance.schemas import CheckinBody, ManualMarkBody
from attendance.services.checkin import perform_checkin

router = APIRouter(tags=["attendance"])


@router.post("/checkin")
async def checkin(body: CheckinBody, db: Database = Depends(get_db),
                  face: FaceClient = Depends(get_face),
                  settings: Settings = Depends(get_settings), user=Depends(current_user)):
    """Authenticated check-in from a student's own device."""
    return await perform_checkin(db, face, settings, b64_to_bytes(body.image),
                                 body.session_id, actor=user)


@router.post("/kiosk/checkin")
async def kiosk_checkin(body: CheckinBody, db: Database = Depends(get_db),
                        face: FaceClient = Depends(get_face),
                        settings: Settings = Depends(get_settings),
                        user=Depends(require_staff)):
    """Kiosk mode — a staff device identifies whoever steps in front of it."""
    return await perform_checkin(db, face, settings, b64_to_bytes(body.image),
                                 body.session_id, actor=user)


@router.post("/attendance/manual")
async def manual_mark(body: ManualMarkBody, db: Database = Depends(get_db),
                      user=Depends(require_staff)):
    if db.session(body.session_id) is None:
        raise HTTPException(404, "Session not found")
    if db.already_marked(body.session_id, body.student_id):
        raise HTTPException(409, "Already marked present")
    db.mark_attendance(body.session_id, body.student_id, "manual", None, user["id"])
    return {"marked": body.student_id, "method": "manual"}


@router.delete("/attendance/{session_id}/{student_id}")
async def unmark(session_id: str, student_id: str, db: Database = Depends(get_db),
                 _=Depends(require_staff)):
    db.unmark_attendance(session_id, student_id)
    return {"unmarked": student_id}
