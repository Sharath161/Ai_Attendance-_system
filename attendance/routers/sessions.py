"""Class session routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from attendance.auth import current_user, require_staff
from attendance.db import Database
from attendance.deps import get_db, user_out
from attendance.schemas import SessionBody

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", status_code=201)
async def open_session(body: SessionBody, db: Database = Depends(get_db),
                       user=Depends(require_staff)):
    if db.one("SELECT 1 FROM courses WHERE id=?", (body.course_id,)) is None:
        raise HTTPException(404, "Course not found")
    if db.open_session_for_course(body.course_id):
        raise HTTPException(409, "This course already has an open session")
    sid = db.open_session(body.course_id, body.title, user["id"])
    return dict(db.session(sid))


@router.post("/{session_id}/close")
async def close_session(session_id: str, db: Database = Depends(get_db),
                        _=Depends(require_staff)):
    if db.session(session_id) is None:
        raise HTTPException(404, "Session not found")
    db.close_session(session_id)
    return dict(db.session(session_id))


@router.get("")
async def list_sessions(course_id: str | None = None, db: Database = Depends(get_db),
                        _=Depends(current_user)):
    return [dict(r) for r in db.list_sessions(course_id)]


@router.get("/{session_id}/attendance")
async def session_attendance(session_id: str, db: Database = Depends(get_db),
                             _=Depends(require_staff)):
    session = db.session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    present = [dict(r) for r in db.session_attendance(session_id)]
    present_ids = {r["student_id"] for r in present}
    absent = [user_out(u) for u in db.course_students(session["course_id"])
              if u["id"] not in present_ids]
    return {"session": dict(session), "present": present, "absent": absent}
