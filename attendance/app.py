"""Attendance system API — built on the separate faceapi recognition service.

Roles
    student : enrol own face, check in, view own attendance
    teacher : open/close sessions for courses, mark manually, view reports
    admin   : everything + manage users, courses, enrolments

All face work (enrol / identify / verify) is delegated over HTTP to faceapi.
"""
from __future__ import annotations

import base64
import csv
import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (Depends, FastAPI, File, Form, HTTPException, Request,
                     UploadFile, status)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field

from attendance import auth
from attendance.auth import (current_user, hash_password, require_admin,
                             require_staff, require_student, verify_password)
from attendance.config import Settings, get_settings
from attendance.db import Database
from attendance.faceclient import FaceClient, FaceServiceError

_WEB = Path(__file__).resolve().parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    app.state.db = Database(s.db_path)
    app.state.face = FaceClient(s)
    # bootstrap an admin so the system is usable on first run
    if not app.state.db.one("SELECT 1 FROM users LIMIT 1"):
        app.state.db.create_user(s.bootstrap_admin_email,
                                 hash_password(s.bootstrap_admin_password),
                                 "System Administrator", "admin")
        print(f"[attendance] bootstrap admin: {s.bootstrap_admin_email} "
              f"/ {s.bootstrap_admin_password}")
    yield
    app.state.db.close()


app = FastAPI(title="Attendance System", version="1.0", lifespan=lifespan,
              description="Face-recognition attendance built on the faceapi service.")

_s = get_settings()
app.add_middleware(CORSMiddleware, allow_origins=_s.cors_list(),
                   allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(HTTPException)
async def _err(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code,
                        content={"error": {"code": exc.status_code,
                                           "message": exc.detail}})


def db(request: Request) -> Database: return request.app.state.db
def face(request: Request) -> FaceClient: return request.app.state.face


def urow(u) -> dict:
    return {"id": u["id"], "email": u["email"], "full_name": u["full_name"],
            "role": u["role"], "student_number": u["student_number"],
            "face_enrolled": bool(u["face_enrolled"])}


async def _read_images(images: list[UploadFile], settings: Settings) -> list[bytes]:
    out = []
    for img in images:
        data = await img.read()
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, f"{img.filename} too large")
        out.append(data)
    return out


def _b64_to_bytes(b64: str) -> bytes:
    if "," in b64[:64]:
        b64 = b64.split(",", 1)[1]
    try:
        return base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 image")


# ── schemas ───────────────────────────────────────────────────────────────────
class LoginBody(BaseModel):
    email: str
    password: str


class RegisterBody(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    full_name: str
    role: str = "student"
    student_number: str | None = None


class CourseBody(BaseModel):
    code: str
    name: str
    teacher_id: str | None = None


class SessionBody(BaseModel):
    course_id: str
    title: str | None = None


class ImageBody(BaseModel):
    image: str                      # base64 / data URI


class EnrolFaceBody(BaseModel):
    images: list[str]               # base64 list


class CheckinBody(ImageBody):
    session_id: str | None = None   # optional: kiosk targeting one session


class ManualMarkBody(BaseModel):
    session_id: str
    student_id: str


# ── auth ──────────────────────────────────────────────────────────────────────
@app.post("/api/auth/login")
async def login(body: LoginBody, request: Request, settings: Settings = Depends(get_settings)):
    user = db(request).user_by_email(body.email)
    if user is None or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    return {"token": auth.create_token(user["id"], user["role"], settings),
            "user": urow(user)}


@app.get("/api/auth/me")
async def me(user=Depends(current_user)):
    return urow(user)


@app.post("/api/users", status_code=201)
async def create_user(body: RegisterBody, request: Request, _=Depends(require_admin)):
    if body.role not in ("student", "teacher", "admin"):
        raise HTTPException(400, "role must be student, teacher or admin")
    if db(request).user_by_email(body.email):
        raise HTTPException(409, "Email already registered")
    uid = db(request).create_user(body.email, hash_password(body.password),
                                  body.full_name, body.role, body.student_number)
    return urow(db(request).user_by_id(uid))


@app.get("/api/users")
async def list_users(request: Request, role: str | None = None, _=Depends(require_staff)):
    return [urow(u) for u in db(request).list_users(role)]


@app.delete("/api/users/{user_id}")
async def delete_user(user_id: str, request: Request, _=Depends(require_admin)):
    u = db(request).user_by_id(user_id)
    if u is None:
        raise HTTPException(404, "User not found")
    await face(request).delete_subject(u["face_subject_id"])
    db(request).run("DELETE FROM users WHERE id=?", (user_id,))
    return {"deleted": user_id}


# ── face enrolment ────────────────────────────────────────────────────────────
@app.post("/api/face/enroll")
async def enroll_face(body: EnrolFaceBody, request: Request,
                      user=Depends(current_user)):
    """Enrol the caller's own face (JSON/base64 — used by the PWA camera)."""
    if not body.images:
        raise HTTPException(400, "At least one image is required")
    images = [_b64_to_bytes(b) for b in body.images]
    try:
        res = await face(request).enroll(user["face_subject_id"], user["full_name"], images)
    except FaceServiceError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(e))
    db(request).mark_face_enrolled(user["id"], True)
    return {"enrolled": True, "embeddings_added": res.get("embeddings_added"),
            "faces_rejected": res.get("faces_rejected")}


@app.post("/api/face/enroll/{user_id}")
async def enroll_face_for(user_id: str, request: Request,
                          images: list[UploadFile] = File(...),
                          settings: Settings = Depends(get_settings),
                          _=Depends(require_staff)):
    """Staff enrol a student from uploaded photo files (multipart)."""
    target = db(request).user_by_id(user_id)
    if target is None:
        raise HTTPException(404, "User not found")
    data = await _read_images(images, settings)
    try:
        res = await face(request).enroll(target["face_subject_id"], target["full_name"], data)
    except FaceServiceError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(e))
    db(request).mark_face_enrolled(user_id, True)
    return {"enrolled": True, "user": urow(db(request).user_by_id(user_id)),
            "embeddings_added": res.get("embeddings_added")}


# ── courses ───────────────────────────────────────────────────────────────────
@app.post("/api/courses", status_code=201)
async def create_course(body: CourseBody, request: Request, _=Depends(require_staff)):
    if db(request).one("SELECT 1 FROM courses WHERE code=?", (body.code.strip(),)):
        raise HTTPException(409, "Course code already exists")
    cid = db(request).create_course(body.code, body.name, body.teacher_id)
    return {"id": cid, "code": body.code, "name": body.name}


@app.get("/api/courses")
async def list_courses(request: Request, user=Depends(current_user)):
    if user["role"] == "student":
        return [dict(r) for r in db(request).courses_for_student(user["id"])]
    return [dict(r) for r in db(request).list_courses()]


@app.post("/api/courses/{course_id}/enrol")
async def enrol_student(course_id: str, student_id: str, request: Request,
                        _=Depends(require_staff)):
    if db(request).user_by_id(student_id) is None:
        raise HTTPException(404, "Student not found")
    db(request).enrol(course_id, student_id)
    return {"enrolled": student_id, "course": course_id}


@app.delete("/api/courses/{course_id}/enrol/{student_id}")
async def unenrol_student(course_id: str, student_id: str, request: Request,
                          _=Depends(require_staff)):
    db(request).unenrol(course_id, student_id)
    return {"unenrolled": student_id}


@app.get("/api/courses/{course_id}/students")
async def course_students(course_id: str, request: Request, _=Depends(require_staff)):
    return [urow(u) for u in db(request).course_students(course_id)]


# ── sessions ──────────────────────────────────────────────────────────────────
@app.post("/api/sessions", status_code=201)
async def open_session(body: SessionBody, request: Request, user=Depends(require_staff)):
    if db(request).one("SELECT 1 FROM courses WHERE id=?", (body.course_id,)) is None:
        raise HTTPException(404, "Course not found")
    existing = db(request).open_session_for_course(body.course_id)
    if existing:
        raise HTTPException(409, "This course already has an open session")
    sid = db(request).open_session(body.course_id, body.title, user["id"])
    return dict(db(request).session(sid))


@app.post("/api/sessions/{session_id}/close")
async def close_session(session_id: str, request: Request, _=Depends(require_staff)):
    if db(request).session(session_id) is None:
        raise HTTPException(404, "Session not found")
    db(request).close_session(session_id)
    return dict(db(request).session(session_id))


@app.get("/api/sessions")
async def list_sessions(request: Request, course_id: str | None = None,
                        _=Depends(current_user)):
    return [dict(r) for r in db(request).list_sessions(course_id)]


@app.get("/api/sessions/{session_id}/attendance")
async def session_attendance(session_id: str, request: Request, _=Depends(require_staff)):
    sess = db(request).session(session_id)
    if sess is None:
        raise HTTPException(404, "Session not found")
    present = [dict(r) for r in db(request).session_attendance(session_id)]
    present_ids = {r["student_id"] for r in present}
    roster = db(request).course_students(sess["course_id"])
    absent = [urow(u) for u in roster if u["id"] not in present_ids]
    return {"session": dict(sess), "present": present, "absent": absent}


# ── check-in (the core loop) ──────────────────────────────────────────────────
async def _do_checkin(request: Request, settings: Settings, image: bytes,
                      session_id: str | None, actor=None):
    """Identify the face, resolve the student, and mark them present."""
    try:
        res = await face(request).identify(image)
    except FaceServiceError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(e))

    if res.get("status") == "no_face":
        return {"status": "no_face", "message": "No face detected — move closer."}
    if res.get("status") == "multiple_faces":
        return {"status": "multiple_faces", "message": "Multiple faces — one person at a time."}
    match = res.get("match")
    if not match:
        best = (res.get("candidates") or [{}])[0].get("score")
        return {"status": "unknown", "message": "Face not recognised.", "best_score": best}

    confidence = float(match.get("score", 0))
    if confidence < settings.min_checkin_confidence:
        return {"status": "low_confidence", "confidence": round(confidence, 4),
                "message": "Recognition confidence too low — try better lighting."}

    student = db(request).user_by_subject(match["subject_id"])
    if student is None:
        return {"status": "unknown_subject",
                "message": "Recognised a face with no matching student record."}

    # resolve which open session to mark
    if session_id:
        sess = db(request).session(session_id)
        if sess is None:
            raise HTTPException(404, "Session not found")
    else:
        rows = [db(request).open_session_for_course(c["id"])
                for c in db(request).courses_for_student(student["id"])]
        open_rows = [r for r in rows if r]
        if not open_rows:
            return {"status": "no_open_session", "student": urow(student),
                    "message": f"No open session for {student['full_name']}'s courses."}
        if len(open_rows) > 1:
            return {"status": "ambiguous_session", "student": urow(student),
                    "message": "Several open sessions — choose one.",
                    "sessions": [dict(r) for r in open_rows]}
        sess = open_rows[0]

    if sess["status"] != "open":
        return {"status": "session_closed", "message": "That session is closed."}
    if not db(request).is_enrolled(sess["course_id"], student["id"]):
        return {"status": "not_enrolled", "student": urow(student),
                "message": f"{student['full_name']} is not enrolled on this course."}

    existing = db(request).already_marked(sess["id"], student["id"])
    if existing:
        return {"status": "already_marked", "student": urow(student),
                "marked_at": existing["marked_at"],
                "message": f"{student['full_name']} is already checked in."}

    db(request).mark_attendance(sess["id"], student["id"], "face",
                                confidence, actor["id"] if actor else None)
    return {"status": "marked", "student": urow(student),
            "confidence": round(confidence, 4), "session_id": sess["id"],
            "message": f"Welcome, {student['full_name']}!"}


@app.post("/api/checkin")
async def checkin(body: CheckinBody, request: Request,
                  settings: Settings = Depends(get_settings), user=Depends(current_user)):
    """Authenticated check-in (student's own phone, or staff-run kiosk)."""
    return await _do_checkin(request, settings, _b64_to_bytes(body.image),
                             body.session_id, actor=user)


@app.post("/api/kiosk/checkin")
async def kiosk_checkin(body: CheckinBody, request: Request,
                        settings: Settings = Depends(get_settings),
                        user=Depends(require_staff)):
    """Kiosk mode — staff device identifies whoever steps in front of it."""
    return await _do_checkin(request, settings, _b64_to_bytes(body.image),
                             body.session_id, actor=user)


@app.post("/api/attendance/manual")
async def manual_mark(body: ManualMarkBody, request: Request, user=Depends(require_staff)):
    sess = db(request).session(body.session_id)
    if sess is None:
        raise HTTPException(404, "Session not found")
    if db(request).already_marked(body.session_id, body.student_id):
        raise HTTPException(409, "Already marked present")
    db(request).mark_attendance(body.session_id, body.student_id, "manual",
                                None, user["id"])
    return {"marked": body.student_id, "method": "manual"}


@app.delete("/api/attendance/{session_id}/{student_id}")
async def unmark(session_id: str, student_id: str, request: Request,
                 _=Depends(require_staff)):
    db(request).unmark_attendance(session_id, student_id)
    return {"unmarked": student_id}


# ── student portal ────────────────────────────────────────────────────────────
@app.get("/api/me/attendance")
async def my_attendance(request: Request, user=Depends(require_student)):
    history = [dict(r) for r in db(request).student_history(user["id"])]
    courses = []
    for r in db(request).student_percentages(user["id"]):
        total, attended = r["total"], r["attended"]
        courses.append({"course_id": r["id"], "code": r["code"], "name": r["name"],
                        "total": total, "attended": attended,
                        "percentage": round(attended / total * 100, 1) if total else 0.0})
    return {"courses": courses, "history": history}


# ── reports ───────────────────────────────────────────────────────────────────
@app.get("/api/reports/course/{course_id}")
async def course_report(course_id: str, request: Request, _=Depends(require_staff)):
    rows = []
    for r in db(request).course_report(course_id):
        total, attended = r["total"], r["attended"]
        rows.append({"student_id": r["id"], "full_name": r["full_name"],
                     "student_number": r["student_number"], "total": total,
                     "attended": attended,
                     "percentage": round(attended / total * 100, 1) if total else 0.0})
    return {"course_id": course_id, "students": rows}


@app.get("/api/reports/course/{course_id}/export.csv")
async def export_csv(course_id: str, request: Request, _=Depends(require_staff)):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["student_number", "full_name", "sessions_attended", "sessions_total", "percentage"])
    for r in db(request).course_report(course_id):
        total, attended = r["total"], r["attended"]
        w.writerow([r["student_number"] or "", r["full_name"], attended, total,
                    round(attended / total * 100, 1) if total else 0.0])
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition":
                                      f'attachment; filename="attendance_{course_id}.csv"'})


@app.get("/api/stats")
async def stats(request: Request, _=Depends(require_staff)):
    return db(request).stats()


# ── health ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health(request: Request):
    face_health = await face(request).health()
    return {"status": "ok", "face_service": face_health,
            "face_service_url": request.app.state.face.base}


# ── PWA (mounted last so /api/* wins) ─────────────────────────────────────────
if _WEB.exists():
    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(_WEB / "index.html", media_type="text/html")

    app.mount("/", StaticFiles(directory=str(_WEB), html=True), name="web")
