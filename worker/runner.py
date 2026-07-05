import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import numpy as np
from sqlalchemy import select, update

from core.config import get_settings
from core.database import SessionLocal
from core.init_db import create_tables_for_local_dev
from core.models import AttendanceEvent, AttendanceStatus, ImageJob, JobStatus, StudentEmbedding
from worker.model_adapter import FaceEmbeddingModel


async def main() -> None:
    settings = get_settings()
    await create_tables_for_local_dev()
    model = FaceEmbeddingModel.from_settings(settings)

    while True:
        processed = await process_next_job(model)
        if not processed:
            await asyncio.sleep(settings.worker_poll_interval_seconds)


async def process_next_job(model: FaceEmbeddingModel) -> bool:
    async with SessionLocal() as session:
        job_result = await session.execute(
            select(ImageJob)
            .where(ImageJob.status == JobStatus.pending)
            .order_by(ImageJob.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        job = job_result.scalar_one_or_none()
        if job is None:
            return False

        job.status = JobStatus.processing
        job.attempts += 1
        await session.commit()

    try:
        embedding, face_count = model.embed_primary_face(Path(job.image_path))
        if face_count == 0 or embedding is None:
            await mark_job_complete(job.id, AttendanceStatus.no_face_detected)
            Path(job.image_path).unlink(missing_ok=True)
            return True
        if face_count > 1:
            await mark_job_complete(job.id, AttendanceStatus.multiple_faces_detected)
            Path(job.image_path).unlink(missing_ok=True)
            return True

        settings = get_settings()
        candidate_id, best_score = await match_student(embedding, settings.worker_model_version)
        threshold = settings.match_threshold
        if candidate_id is None:
            # No enrolled embeddings to compare against.
            await create_attendance_event(job.id, None, None, AttendanceStatus.unknown)
        elif best_score >= threshold:
            await create_attendance_event(
                job.id, candidate_id, best_score, AttendanceStatus.recognized, candidate_id
            )
        else:
            # A candidate existed but did not clear the threshold; record the guess
            # for review without attributing attendance to the student.
            await create_attendance_event(
                job.id, None, best_score, AttendanceStatus.low_confidence, candidate_id
            )
        Path(job.image_path).unlink(missing_ok=True)
        return True
    except Exception as exc:
        await mark_job_failed(job.id, str(exc))
        return True


async def match_student(
    query_embedding: np.ndarray,
    model_version: str | None = None,
) -> tuple[UUID | None, float]:
    """Return the closest enrolled student and the cosine score.

    Only embeddings whose ``model_version`` matches ``model_version`` are
    considered; this prevents dimension mismatches when the pipeline is
    upgraded (e.g. SFace 128-d → ArcFace 512-d).  Pass ``None`` to skip the
    filter (legacy behaviour).

    The student id is ``None`` only when there are no active embeddings to
    compare against; the caller applies the match threshold so it can tell a
    sub-threshold candidate (``low_confidence``) apart from no candidate at all
    (``unknown``).
    """
    async with SessionLocal() as session:
        q = (
            select(StudentEmbedding.student_id, StudentEmbedding.embedding)
            .where(StudentEmbedding.active.is_(True))
        )
        if model_version:
            q = q.where(StudentEmbedding.model_version == model_version)
        result = await session.execute(q)
        best_student_id: UUID | None = None
        best_score = -1.0

        for student_id, stored_embedding in result.all():
            candidate = np.asarray(stored_embedding, dtype=np.float32)
            score = float(np.dot(query_embedding, candidate))
            if score > best_score:
                best_student_id = student_id
                best_score = score

        return best_student_id, best_score


async def create_attendance_event(
    job_id: UUID,
    student_id: UUID | None,
    confidence: float | None,
    attendance_status: AttendanceStatus,
    candidate_student_id: UUID | None = None,
) -> None:
    async with SessionLocal() as session:
        job = await session.get(ImageJob, job_id)
        if job is None:
            return

        session.add(
            AttendanceEvent(
                student_id=student_id,
                candidate_student_id=candidate_student_id,
                job_id=job.id,
                camera_id=job.camera_id,
                class_id=job.class_id,
                captured_at=job.captured_at,
                confidence=confidence,
                status=attendance_status,
            )
        )
        job.status = JobStatus.complete
        job.processed_at = datetime.now(timezone.utc)
        await session.commit()


async def mark_job_complete(job_id: UUID, attendance_status: AttendanceStatus) -> None:
    async with SessionLocal() as session:
        job = await session.get(ImageJob, job_id)
        if job is None:
            return

        session.add(
            AttendanceEvent(
                student_id=None,
                job_id=job.id,
                camera_id=job.camera_id,
                class_id=job.class_id,
                captured_at=job.captured_at,
                confidence=None,
                status=attendance_status,
            )
        )
        job.status = JobStatus.complete
        job.processed_at = datetime.now(timezone.utc)
        await session.commit()


async def mark_job_failed(job_id: UUID, message: str) -> None:
    async with SessionLocal() as session:
        await session.execute(
            update(ImageJob)
            .where(ImageJob.id == job_id)
            .values(
                status=JobStatus.failed,
                error_message=message[:1000],
                processed_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()


if __name__ == "__main__":
    asyncio.run(main())
