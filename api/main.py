from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import cv2
import numpy as np
import aiofiles
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import Settings, get_settings
from core.database import get_session
from core.init_db import create_tables_for_local_dev
from core.models import (
    AttendanceEvent,
    AttendanceStatus,
    DeviceStatus,
    ESP32Device,
    ImageJob,
    JobStatus,
    Student,
    StudentEmbedding,
    StudentStatus,
)
from worker.model_adapter import FaceEmbeddingModel

_PAGES = Path(__file__).parent

CHECKIN_COOLDOWN_MINUTES = 5


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_tables_for_local_dev()
    settings = get_settings()
    try:
        app.state.face_model = FaceEmbeddingModel.from_settings(settings)
        print(f"[startup] face model loaded: {settings.worker_model_version}")
    except FileNotFoundError as exc:
        app.state.face_model = None
        print(f"[startup] face model NOT loaded ({exc}) — run python -m worker.download_models")
    yield


app = FastAPI(title="AI Face Attendance API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── page routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse, include_in_schema=False)
async def landing_page() -> FileResponse:
    return FileResponse(_PAGES / "index.html", media_type="text/html")


@app.get("/register/webcam", response_class=FileResponse, include_in_schema=False)
async def webcam_register_page() -> FileResponse:
    return FileResponse(_PAGES / "webcam_register.html", media_type="text/html")


@app.get("/checkin", response_class=FileResponse, include_in_schema=False)
async def checkin_page() -> FileResponse:
    return FileResponse(_PAGES / "checkin.html", media_type="text/html")


@app.get("/student", response_class=FileResponse, include_in_schema=False)
async def student_page() -> FileResponse:
    return FileResponse(_PAGES / "student.html", media_type="text/html")


@app.get("/admin", response_class=FileResponse, include_in_schema=False)
async def admin_page() -> FileResponse:
    return FileResponse(_PAGES / "admin.html", media_type="text/html")


# ── health / metrics ──────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "face_model_loaded": app.state.face_model is not None,
    }


@app.get("/metrics/backlog")
async def backlog_metrics(
    session: AsyncSession = Depends(get_session),
    average_processing_ms: int = 150,
) -> dict:
    pending_count = await session.scalar(
        select(func.count()).select_from(ImageJob).where(ImageJob.status == JobStatus.pending)
    )
    oldest_pending = await session.scalar(
        select(func.min(ImageJob.created_at)).where(ImageJob.status == JobStatus.pending)
    )
    pending_jobs = int(pending_count or 0)
    estimated_seconds = pending_jobs * (average_processing_ms / 1000)
    oldest_age_seconds = None
    if oldest_pending is not None:
        oldest_age_seconds = (datetime.now(timezone.utc) - oldest_pending).total_seconds()
    return {
        "pending_jobs": pending_jobs,
        "average_processing_ms": average_processing_ms,
        "estimated_catch_up_seconds": estimated_seconds,
        "oldest_pending_age_seconds": oldest_age_seconds,
    }


# ── ESP32 device management ───────────────────────────────────────────────────

@app.post("/devices/register", status_code=status.HTTP_201_CREATED)
async def register_device(
    name: str = Form(...),
    room_label: str | None = Form(None),
    device_mac: str | None = Form(None),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Admin endpoint: register a new ESP32-CAM unit. Returns the API key to flash onto the device."""
    if device_mac:
        existing = await session.scalar(
            select(ESP32Device).where(ESP32Device.device_mac == device_mac)
        )
        if existing:
            raise HTTPException(409, detail="Device MAC already registered")

    device = ESP32Device(name=name, room_label=room_label, device_mac=device_mac)
    session.add(device)
    await session.commit()
    return {
        "device_id": str(device.id),
        "name": device.name,
        "room_label": device.room_label,
        "api_key": device.api_key,   # show once — store securely
        "status": device.status.value,
    }


@app.get("/devices")
async def list_devices(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """Admin: list all registered ESP32 devices and their last-seen status."""
    result = await session.execute(select(ESP32Device).order_by(ESP32Device.name))
    return [
        {
            "device_id": str(d.id),
            "name": d.name,
            "room_label": d.room_label,
            "device_mac": d.device_mac,
            "firmware_version": d.firmware_version,
            "last_seen_at": d.last_seen_at.isoformat() if d.last_seen_at else None,
            "status": d.status.value,
        }
        for d in result.scalars()
    ]


@app.get("/devices/{device_id}")
async def get_device(device_id: UUID, session: AsyncSession = Depends(get_session)) -> dict:
    device = await session.get(ESP32Device, device_id)
    if not device:
        raise HTTPException(404, detail="Device not found")
    return {
        "device_id": str(device.id),
        "name": device.name,
        "room_label": device.room_label,
        "device_mac": device.device_mac,
        "firmware_version": device.firmware_version,
        "last_seen_at": device.last_seen_at.isoformat() if device.last_seen_at else None,
        "status": device.status.value,
        "created_at": device.created_at.isoformat(),
    }


# ── ESP32 image ingest (the endpoint the ESP32 calls) ─────────────────────────

async def _authenticate_device(
    x_device_key: str | None,
    session: AsyncSession,
) -> ESP32Device:
    """Validate X-Device-Key header and return the matching ESP32Device."""
    if not x_device_key:
        raise HTTPException(status_code=401, detail="Missing X-Device-Key header")
    device = await session.scalar(
        select(ESP32Device).where(
            ESP32Device.api_key == x_device_key,
            ESP32Device.status == DeviceStatus.active,
        )
    )
    if not device:
        raise HTTPException(status_code=403, detail="Invalid or inactive device key")
    return device


@app.post("/ingest/image", status_code=status.HTTP_202_ACCEPTED)
async def ingest_image(
    image: UploadFile = File(...),
    class_id: str | None = Form(None),
    captured_at: datetime | None = Form(None),
    firmware_version: str | None = Form(None),
    x_device_key: str | None = Header(default=None),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict:
    """
    Primary endpoint called by every ESP32-CAM unit.
    Authenticates the device via X-Device-Key header, saves the JPEG to disk,
    and enqueues an ImageJob for the batch worker to process.
    """
    device = await _authenticate_device(x_device_key, session)

    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, detail="Only JPEG/PNG/WebP images accepted")

    event_time = captured_at or datetime.now(timezone.utc)

    # Stream image to spool directory: work/spool/YYYY/MM/DD/HH/<uuid>.jpg
    image_path = await _stream_to_spool(image, event_time, settings)

    # Update device heartbeat and firmware version
    now = datetime.now(timezone.utc)
    device.last_seen_at = now
    if firmware_version:
        device.firmware_version = firmware_version

    job = ImageJob(
        image_path=str(image_path),
        camera_id=device.name,
        device_id=device.id,
        class_id=class_id,
        captured_at=event_time,
        status=JobStatus.pending,
    )
    session.add(job)
    await session.commit()

    return {
        "job_id": str(job.id),
        "status": "queued",
        "device": device.name,
        "image_path": str(image_path),
        "captured_at": event_time.isoformat(),
    }


@app.get("/ingest/queue/status")
async def queue_status(session: AsyncSession = Depends(get_session)) -> dict:
    """Quick status summary of the image processing queue."""
    counts = {}
    for s in JobStatus:
        counts[s.value] = int(await session.scalar(
            select(func.count()).select_from(ImageJob).where(ImageJob.status == s)
        ) or 0)

    oldest = await session.scalar(
        select(func.min(ImageJob.captured_at)).where(ImageJob.status == JobStatus.pending)
    )
    return {
        "queue": counts,
        "oldest_pending_captured_at": oldest.isoformat() if oldest else None,
    }


# ── attendance capture (async queue path) ─────────────────────────────────────

@app.post("/attendance/capture", status_code=status.HTTP_202_ACCEPTED)
async def capture_attendance(
    camera_id: str = Form(...),
    class_id: str | None = Form(None),
    captured_at: datetime | None = Form(None),
    image: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict:
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Unsupported image format")
    event_time = captured_at or datetime.now(timezone.utc)
    image_path = await _stream_to_spool(image, event_time, settings)
    job = ImageJob(
        image_path=str(image_path),
        camera_id=camera_id,
        class_id=class_id,
        captured_at=event_time,
    )
    session.add(job)
    await session.commit()
    return {"job_id": str(job.id), "status": "accepted"}


@app.get("/attendance/events")
async def attendance_events(
    limit: int = 100,
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    limit = min(max(limit, 1), 500)
    result = await session.execute(
        select(
            AttendanceEvent.id,
            AttendanceEvent.student_id,
            AttendanceEvent.candidate_student_id,
            AttendanceEvent.camera_id,
            AttendanceEvent.class_id,
            AttendanceEvent.captured_at,
            AttendanceEvent.confidence,
            AttendanceEvent.status,
            AttendanceEvent.created_at,
        )
        .order_by(AttendanceEvent.created_at.desc())
        .limit(limit)
    )
    return [
        {
            "id": str(row.id),
            "student_id": str(row.student_id) if row.student_id else None,
            "candidate_student_id": str(row.candidate_student_id) if row.candidate_student_id else None,
            "camera_id": row.camera_id,
            "class_id": row.class_id,
            "captured_at": row.captured_at.isoformat(),
            "confidence": row.confidence,
            "status": row.status.value,
            "created_at": row.created_at.isoformat(),
        }
        for row in result.all()
    ]


# ── check-in (inline synchronous recognition) ─────────────────────────────────

@app.post("/checkin/recognize")
async def checkin_recognize(
    image: UploadFile = File(...),
    camera_id: str = Form("checkin-kiosk"),
    class_id: str | None = Form(None),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict:
    model: FaceEmbeddingModel | None = app.state.face_model
    if model is None:
        raise HTTPException(503, detail="Face model not loaded. Run: python -m worker.download_models")

    data = await image.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, detail="Could not decode image")

    embedding, face_count, bbox = model.detect_and_embed(frame)

    if face_count == 0 or embedding is None:
        return {"status": "no_face", "face_count": 0, "face_box": None}

    if face_count > 1:
        return {"status": "multiple_faces", "face_count": face_count, "face_box": bbox}

    candidate_id, best_score, demographic_group = await _match_student(
        session, embedding, settings.worker_model_version
    )

    if candidate_id is None:
        return {
            "status": "unknown", "face_count": 1, "face_box": bbox,
            "confidence": None, "name": None, "student_ext_id": None,
            "demographic_group": None, "threshold_used": settings.match_threshold,
        }

    # Use per-demographic threshold if configured, else global
    threshold = settings.threshold_for(demographic_group)

    if best_score < threshold:
        candidate = await session.get(Student, candidate_id)
        return {
            "status": "low_confidence", "face_count": 1, "face_box": bbox,
            "confidence": round(float(best_score), 4),
            "name": candidate.name if candidate else None,
            "student_ext_id": candidate.student_id if candidate else None,
            "demographic_group": demographic_group,
            "threshold_used": round(threshold, 4),
        }

    student = await session.get(Student, candidate_id)
    already_in = await _checked_in_recently(session, candidate_id, CHECKIN_COOLDOWN_MINUTES)

    event_id: str | None = None
    if not already_in:
        now = datetime.now(timezone.utc)
        job = ImageJob(
            image_path=f"checkin:live:{now.isoformat()}",
            camera_id=camera_id,
            class_id=class_id,
            captured_at=now,
            status=JobStatus.complete,
            processed_at=now,
        )
        session.add(job)
        await session.flush()
        event = AttendanceEvent(
            student_id=candidate_id,
            candidate_student_id=candidate_id,
            job_id=job.id,
            camera_id=camera_id,
            class_id=class_id,
            captured_at=now,
            confidence=best_score,
            status=AttendanceStatus.recognized,
        )
        session.add(event)
        await session.commit()
        event_id = str(event.id)

    return {
        "status": "recognized",
        "face_count": 1,
        "face_box": bbox,
        "confidence": round(float(best_score), 4),
        "name": student.name if student else None,
        "student_ext_id": student.student_id if student else None,
        "already_checked_in": already_in,
        "event_id": event_id,
        "demographic_group": demographic_group,
        "threshold_used": round(threshold, 4),
    }


# ── student list & attendance queries (student portal) ───────────────────────

@app.get("/students/list")
async def list_students(session: AsyncSession = Depends(get_session)) -> list[dict]:
    result = await session.execute(select(Student).order_by(Student.name))
    return [
        {
            "student_id": s.student_id,
            "name": s.name,
            "class_id": s.class_id,
            "status": s.status.value,
        }
        for s in result.scalars()
    ]


@app.get("/students/{student_id}/attendance/summary")
async def student_attendance_summary(
    student_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    student = await session.scalar(select(Student).where(Student.student_id == student_id))
    if not student:
        raise HTTPException(404, detail="Student not found")

    # Distinct class_ids that have any attendance events
    class_ids_result = await session.execute(
        select(AttendanceEvent.class_id).distinct()
        .where(AttendanceEvent.class_id.isnot(None))
    )
    class_ids = [r[0] for r in class_ids_result.all() if r[0]]

    modules = []
    total_attended = 0
    total_sessions_all = 0

    for cid in class_ids:
        # Count total events for this class as a proxy for "total sessions"
        total = int(await session.scalar(
            select(func.count()).select_from(AttendanceEvent)
            .where(AttendanceEvent.class_id == cid)
        ) or 0)

        # Student's recognized events in this class
        attended = int(await session.scalar(
            select(func.count()).select_from(AttendanceEvent)
            .where(
                AttendanceEvent.class_id == cid,
                AttendanceEvent.student_id == student.id,
                AttendanceEvent.status == AttendanceStatus.recognized,
            )
        ) or 0)

        pct = (attended / total * 100) if total > 0 else 0.0
        modules.append({
            "class_id": cid,
            "code": cid,
            "name": _friendly_class_name(cid),
            "total": total,
            "attended": attended,
            "pct": round(pct, 1),
        })
        total_attended += attended
        total_sessions_all += total

    overall_pct = (total_attended / total_sessions_all * 100) if total_sessions_all > 0 else 0.0

    return {
        "student_id": student_id,
        "name": student.name,
        "overall_pct": round(overall_pct, 1),
        "total_sessions": total_sessions_all,
        "modules": sorted(modules, key=lambda m: m["code"]),
    }


@app.get("/students/{student_id}/attendance/sessions")
async def student_attendance_sessions(
    student_id: str,
    class_id: str,
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    student = await session.scalar(select(Student).where(Student.student_id == student_id))
    if not student:
        raise HTTPException(404, detail="Student not found")

    result = await session.execute(
        select(AttendanceEvent)
        .where(
            AttendanceEvent.student_id == student.id,
            AttendanceEvent.class_id == class_id,
        )
        .order_by(AttendanceEvent.captured_at.desc())
        .limit(50)
    )
    return [
        {
            "captured_at": e.captured_at.isoformat(),
            "status": e.status.value,
            "confidence": e.confidence,
            "camera_id": e.camera_id,
            "method": "manual" if (e.camera_id or "").startswith("manual") else "face_recognition",
        }
        for e in result.scalars()
    ]


# ── admin endpoints ───────────────────────────────────────────────────────────

@app.get("/admin/fairness")
async def admin_fairness(session: AsyncSession = Depends(get_session)) -> dict:
    """Per-demographic recognition stats for the fairness KPI panel."""
    settings = get_settings()

    # Aggregate attendance events joined with student demographic labels
    # We use the demographic_group stored on StudentEmbedding as the source of truth
    result = await session.execute(
        select(
            StudentEmbedding.demographic_group,
            AttendanceEvent.status,
            func.count().label("cnt"),
        )
        .join(AttendanceEvent, AttendanceEvent.student_id == StudentEmbedding.student_id)
        .where(StudentEmbedding.demographic_group.isnot(None))
        .group_by(StudentEmbedding.demographic_group, AttendanceEvent.status)
    )

    demos: dict[str, dict] = {}
    for group, status_val, cnt in result.all():
        if group not in demos:
            demos[group] = {"recognized": 0, "unknown": 0, "low_confidence": 0, "total": 0}
        key = status_val.value if hasattr(status_val, "value") else str(status_val)
        if key in demos[group]:
            demos[group][key] += cnt
        demos[group]["total"] += cnt

    for g, d in demos.items():
        d["rate"] = round(d["recognized"] / d["total"], 4) if d["total"] > 0 else 0.0

    return {
        "demographics": demos,
        "thresholds": settings.race_thresholds,
    }


@app.post("/attendance/manual", status_code=status.HTTP_201_CREATED)
async def manual_attendance(
    student_id: str = Form(...),
    class_id: str = Form(...),
    captured_at: datetime | None = Form(None),
    status_val: str = Form("recognized", alias="status"),
    notes: str | None = Form(None),
    camera_id: str = Form("manual-override"),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Admin manual attendance override — bypasses face recognition."""
    student = await session.scalar(select(Student).where(Student.student_id == student_id))
    if not student:
        raise HTTPException(404, detail="Student not found")

    try:
        att_status = AttendanceStatus(status_val)
    except ValueError:
        raise HTTPException(400, detail=f"Invalid status: {status_val}")

    event_time = captured_at or datetime.now(timezone.utc)

    job = ImageJob(
        image_path=f"manual:{notes or ''}",
        camera_id=camera_id,
        class_id=class_id,
        captured_at=event_time,
        status=JobStatus.complete,
        processed_at=datetime.now(timezone.utc),
    )
    session.add(job)
    await session.flush()

    event = AttendanceEvent(
        student_id=student.id,
        candidate_student_id=student.id,
        job_id=job.id,
        camera_id=camera_id,
        class_id=class_id,
        captured_at=event_time,
        confidence=1.0,
        status=att_status,
    )
    session.add(event)
    await session.commit()
    return {"event_id": str(event.id), "status": att_status.value}


# ── Hypertuning / optimization ────────────────────────────────────────────────

_optimization_running: bool = False
_last_optimization_result: dict | None = None


@app.post("/admin/optimize", status_code=status.HTTP_202_ACCEPTED)
async def trigger_optimization(
    background_tasks: BackgroundTasks,
    min_quality: float = 0.15,
) -> dict:
    """Run the 3-stage embedding optimization pipeline in the background.

    Stage 1 — Quality scoring (sharpness + YuNet confidence)
    Stage 2 — Weighted prototype per student
    Stage 3 — Leave-one-out threshold calibration + per-demographic tuning
    """
    global _optimization_running
    if _optimization_running:
        raise HTTPException(409, detail="Optimization already running")
    _optimization_running = True
    background_tasks.add_task(_run_optimization_task, min_quality)
    return {"status": "started", "min_quality": min_quality}


@app.get("/admin/optimize/status")
async def optimization_status() -> dict:
    return {
        "running": _optimization_running,
        "last_result": _last_optimization_result,
    }


def _run_optimization_task(min_quality: float) -> None:
    import asyncio
    from worker.optimizer import run_optimization, _write_thresholds_to_env
    from pathlib import Path
    global _optimization_running, _last_optimization_result
    try:
        result = asyncio.run(run_optimization(min_quality=min_quality))
        _last_optimization_result = {
            "global_threshold":     result.global_threshold,
            "race_thresholds":      result.race_thresholds,
            "global_f1":            result.global_f1,
            "students_processed":   result.students_processed,
            "embeddings_scored":    result.embeddings_scored,
            "prototypes_built":     result.prototypes_built,
            "per_demographic_f1":   result.per_demographic_f1,
            "low_quality_count":    result.low_quality_count,
            "completed_at":         datetime.now(timezone.utc).isoformat(),
        }
        # Patch .env so thresholds survive restart
        root = Path(__file__).parent.parent
        _write_thresholds_to_env(
            result.global_threshold,
            result.race_thresholds,
            env_path=root / ".env",
        )
        print(f"[optimize] Done — global_t={result.global_threshold}  F1={result.global_f1}")
    except Exception as exc:
        _last_optimization_result = {"error": str(exc)}
        print(f"[optimize] ERROR: {exc}")
    finally:
        _optimization_running = False


def _friendly_class_name(class_id: str) -> str:
    """Best-effort human-readable name from a class_id like CS301 or CS301-A."""
    prefixes = {
        "CS": "Computer Science",
        "SE": "Software Engineering",
        "AI": "Artificial Intelligence",
        "DS": "Data Science",
        "DB": "Database Systems",
        "NW": "Networking",
        "MA": "Mathematics",
        "EC": "Electronics",
    }
    for prefix, name in prefixes.items():
        if class_id.upper().startswith(prefix):
            return f"{name} ({class_id})"
    return class_id


# ── student registration ──────────────────────────────────────────────────────

@app.post("/students/register", status_code=status.HTTP_202_ACCEPTED)
async def register_student(
    student_id: str = Form(...),
    name: str = Form(...),
    class_id: str | None = Form(None),
    # Optional demographic label for fairness tracking.
    # Expected values: white, black, indian, east_asian, southeast_asian,
    # middle_eastern, latino_hispanic  (or any free-form string)
    demographic_group: str | None = Form(None),
    images: list[UploadFile] = File(...),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict:
    import json

    existing = await session.scalar(select(Student).where(Student.student_id == student_id))
    if existing is not None:
        raise HTTPException(status_code=409, detail="Student ID already exists")

    student = Student(
        student_id=student_id,
        name=name,
        class_id=class_id,
        status=StudentStatus.pending_embedding,
    )
    session.add(student)

    reg_dir = settings.spool_dir.parent / "registrations" / student_id
    reg_dir.mkdir(parents=True, exist_ok=True)

    # Persist optional metadata so the embedding worker can read it
    if demographic_group:
        (reg_dir / "metadata.json").write_text(
            json.dumps({"demographic_group": demographic_group}), encoding="utf-8"
        )

    saved = 0
    for image in images:
        if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            continue
        suffix = Path(image.filename or "").suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"
        await _stream_to_path(image, reg_dir / f"{uuid4()}{suffix}", settings.max_upload_bytes)
        saved += 1

    if saved == 0:
        raise HTTPException(status_code=400, detail="At least one valid image is required")

    await session.commit()
    return {
        "student_id": student_id,
        "status": "pending_embedding",
        "images_saved": saved,
        "demographic_group": demographic_group,
    }


@app.post("/students/{student_id}/process")
async def process_student(
    student_id: str,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> dict:
    student = await session.scalar(select(Student).where(Student.student_id == student_id))
    if student is None:
        raise HTTPException(404, detail="Student not found")
    if student.status == StudentStatus.active:
        emb_count = await session.scalar(
            select(func.count()).select_from(StudentEmbedding)
            .where(StudentEmbedding.student_id == student.id, StudentEmbedding.active.is_(True))
        )
        return {"status": "active", "embeddings": int(emb_count or 0)}
    background_tasks.add_task(_run_registration_updater)
    return {"status": "processing"}


@app.get("/students/{student_id}/status")
async def student_status(
    student_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    student = await session.scalar(select(Student).where(Student.student_id == student_id))
    if student is None:
        raise HTTPException(404, detail="Student not found")
    emb_count = await session.scalar(
        select(func.count()).select_from(StudentEmbedding)
        .where(StudentEmbedding.student_id == student.id, StudentEmbedding.active.is_(True))
    )
    return {
        "student_id": student_id,
        "status": student.status.value,
        "active_embeddings": int(emb_count or 0),
    }


# ── internal helpers ──────────────────────────────────────────────────────────

async def _match_student(
    session: AsyncSession,
    query: np.ndarray,
    model_version: str | None = None,
) -> tuple[UUID | None, float, str | None]:
    """Return (student_id, best_score, demographic_group) for the closest embedding match."""
    q = select(
        StudentEmbedding.student_id,
        StudentEmbedding.embedding,
        StudentEmbedding.demographic_group,
    ).where(StudentEmbedding.active.is_(True))
    if model_version:
        q = q.where(StudentEmbedding.model_version == model_version)
    result = await session.execute(q)
    best_id: UUID | None = None
    best_score = -1.0
    best_demographic: str | None = None
    for sid, emb, demo in result.all():
        score = float(np.dot(query, np.asarray(emb, dtype=np.float32)))
        if score > best_score:
            best_id = sid
            best_score = score
            best_demographic = demo
    return best_id, best_score, best_demographic


async def _checked_in_recently(
    session: AsyncSession, student_id: UUID, minutes: int
) -> bool:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    count = await session.scalar(
        select(func.count()).select_from(AttendanceEvent)
        .where(
            AttendanceEvent.student_id == student_id,
            AttendanceEvent.status == AttendanceStatus.recognized,
            AttendanceEvent.created_at >= cutoff,
        )
    )
    return (count or 0) > 0


async def _stream_to_spool(image: UploadFile, captured_at: datetime, settings: Settings) -> Path:
    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    directory = settings.spool_dir / captured_at.strftime("%Y/%m/%d/%H")
    directory.mkdir(parents=True, exist_ok=True)
    final_path = directory / f"{uuid4()}{suffix}"
    await _stream_to_path(image, final_path, settings.max_upload_bytes)
    return final_path


async def _stream_to_path(image: UploadFile, final_path: Path, max_bytes: int) -> None:
    tmp = final_path.with_suffix(f"{final_path.suffix}.partial")
    written = 0
    try:
        async with aiofiles.open(tmp, "wb") as out:
            while chunk := await image.read(256 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(413, detail="Image exceeds upload limit")
                await out.write(chunk)
        tmp.replace(final_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _run_registration_updater() -> None:
    import asyncio
    from worker.registration_updater import process_pending_registrations
    asyncio.run(process_pending_registrations())
