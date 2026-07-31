"""Check-in business logic.

Kept out of the routers so the rules (which session, is the student enrolled,
have they already been marked, is confidence sufficient) are testable in
isolation and shared by both the student and kiosk entry points.
"""
from __future__ import annotations

from fastapi import HTTPException, status

from attendance.config import Settings
from attendance.db import Database
from attendance.deps import user_out
from attendance.faceclient import FaceClient, FaceServiceError


async def perform_checkin(
    db: Database,
    face: FaceClient,
    settings: Settings,
    image: bytes,
    session_id: str | None = None,
    actor: dict | None = None,
) -> dict:
    """Identify a face and mark the resulting student present.

    Returns a status dict rather than raising for business outcomes, so callers
    (and the UI) can distinguish 'unknown face' from 'already marked' etc.
    """
    try:
        result = await face.identify(image)
    except FaceServiceError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc))

    # ── recognition outcomes ─────────────────────────────────────────────────
    if result.get("status") == "no_face":
        return {"status": "no_face", "message": "No face detected — move closer."}
    if result.get("status") == "multiple_faces":
        return {"status": "multiple_faces",
                "message": "Multiple faces detected — one person at a time."}

    match = result.get("match")
    if not match:
        best = (result.get("candidates") or [{}])[0].get("score")
        return {"status": "unknown", "message": "Face not recognised.", "best_score": best}

    confidence = float(match.get("score", 0.0))
    if confidence < settings.min_checkin_confidence:
        return {"status": "low_confidence", "confidence": round(confidence, 4),
                "message": "Recognition confidence too low — try better lighting."}

    student = db.user_by_subject(match["subject_id"])
    if student is None:
        return {"status": "unknown_subject",
                "message": "Recognised a face with no matching student record."}

    # ── which session are we marking? ────────────────────────────────────────
    session = _resolve_session(db, student, session_id)
    if isinstance(session, dict):        # an error/status payload
        return session

    if session["status"] != "open":
        return {"status": "session_closed", "message": "That session is closed."}

    if not db.is_enrolled(session["course_id"], student["id"]):
        return {"status": "not_enrolled", "student": user_out(student),
                "message": f"{student['full_name']} is not enrolled on this course."}

    existing = db.already_marked(session["id"], student["id"])
    if existing:
        return {"status": "already_marked", "student": user_out(student),
                "marked_at": existing["marked_at"],
                "message": f"{student['full_name']} is already checked in."}

    db.mark_attendance(session["id"], student["id"], "face", confidence,
                       actor["id"] if actor else None)
    return {"status": "marked", "student": user_out(student),
            "confidence": round(confidence, 4), "session_id": session["id"],
            "message": f"Welcome, {student['full_name']}!"}


def _resolve_session(db: Database, student, session_id: str | None):
    """Return the session row to mark, or a status dict describing why we can't."""
    if session_id:
        session = db.session(session_id)
        if session is None:
            raise HTTPException(404, "Session not found")
        return session

    open_sessions = [s for s in
                     (db.open_session_for_course(c["id"])
                      for c in db.courses_for_student(student["id"])) if s]

    if not open_sessions:
        return {"status": "no_open_session", "student": user_out(student),
                "message": f"No open session for {student['full_name']}'s courses."}
    if len(open_sessions) > 1:
        return {"status": "ambiguous_session", "student": user_out(student),
                "message": "Several open sessions — choose one.",
                "sessions": [dict(s) for s in open_sessions]}
    return open_sessions[0]
