"""Seed the database with synthetic students and attendance history.

Creates realistic load for dashboard testing without needing real photos.
Each student gets a unique deterministic 128-d embedding (seeded from their
UUID), so recognition queries against their own embedding score 1.0 and
queries against other students score near 0 — the same behaviour you'd get
from a real face model.

Usage
-----
    # default: 100 students, 5 classes, 7 days of history
    python -m tests.stress.seed_data

    # custom scale
    python -m tests.stress.seed_data --students 200 --classes 8 --days 14

    # wipe previous seed data first
    python -m tests.stress.seed_data --reset
"""
from __future__ import annotations

import argparse
import asyncio
import random
import time
import uuid
from datetime import datetime, timedelta, timezone
from itertools import cycle

import numpy as np
from sqlalchemy import delete, select

from core.config import get_settings
from core.database import SessionLocal
from core.init_db import create_tables_for_local_dev
from core.models import (
    AttendanceEvent,
    AttendanceStatus,
    ImageJob,
    JobStatus,
    Student,
    StudentEmbedding,
    StudentStatus,
)
from core.math_utils import l2_normalize

# ── name pool ─────────────────────────────────────────────────────────────────

FIRST = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn",
    "Skyler", "Dakota", "Reese", "Finley", "Cameron", "Sage", "Blake", "Hayden",
    "Kendall", "Logan", "Parker", "Drew", "Rowan", "Emery", "Peyton", "Spencer",
    "Elliot", "Indira", "Priya", "Wei", "Mei", "Luca", "Mateo", "Sofia",
    "Amara", "Kofi", "Nia", "Yuki", "Hana", "Tariq", "Zara", "Felix",
]
LAST = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson",
    "White", "Harris", "Martin", "Thompson", "Lee", "Walker", "Hall", "Young",
    "Allen", "King", "Scott", "Green", "Baker", "Adams", "Nelson", "Carter",
    "Patel", "Khan", "Singh", "Nguyen", "Kim", "Tanaka", "Okafor", "Santos",
    "Rossi", "Müller",
]

CLASSES = ["CS-101", "CS-102", "MATH-201", "PHY-301", "ENG-101",
           "BIO-110", "CHEM-201", "ECON-101", "HIST-201", "ART-101"]

# mix of camera IDs to make the log realistic
CAMERAS = ["cam-entrance-A", "cam-entrance-B", "cam-hall-1", "cam-hall-2", "cam-lab-1"]

# attendance event status distribution (weights)
EVENT_WEIGHTS = {
    AttendanceStatus.recognized:             0.78,
    AttendanceStatus.low_confidence:         0.08,
    AttendanceStatus.unknown:                0.07,
    AttendanceStatus.no_face_detected:       0.05,
    AttendanceStatus.multiple_faces_detected: 0.02,
}
STATUSES, WEIGHTS = zip(*EVENT_WEIGHTS.items())


# ── embedding helpers ─────────────────────────────────────────────────────────

def make_embedding(seed: uuid.UUID, dims: int = 128) -> list[float]:
    rng = np.random.default_rng(seed.int & 0xFFFF_FFFF_FFFF_FFFF)
    v = rng.normal(size=dims).astype(np.float32)
    return l2_normalize(v).tolist()


def perturb(embedding: list[float], noise: float = 0.15) -> list[float]:
    v = np.asarray(embedding, dtype=np.float32)
    v += np.random.default_rng().normal(0, noise, size=v.shape).astype(np.float32)
    return l2_normalize(v).tolist()


def random_embedding(dims: int = 128) -> list[float]:
    v = np.random.default_rng().normal(size=dims).astype(np.float32)
    return l2_normalize(v).tolist()


# ── session distribution ──────────────────────────────────────────────────────

def session_times(day: datetime) -> list[datetime]:
    """Return two check-in windows per day (morning + afternoon)."""
    return [
        day.replace(hour=8,  minute=0),
        day.replace(hour=13, minute=0),
    ]


def jitter(base: datetime, spread_minutes: int = 45) -> datetime:
    return base + timedelta(seconds=random.randint(0, spread_minutes * 60))


# ── main seeder ───────────────────────────────────────────────────────────────

async def reset_seed_data() -> None:
    async with SessionLocal() as s:
        await s.execute(delete(AttendanceEvent))
        await s.execute(delete(ImageJob))
        await s.execute(delete(StudentEmbedding))
        await s.execute(delete(Student))
        await s.commit()
    print("reset: all students and events cleared")


async def seed(n_students: int, n_classes: int, n_days: int, model_version: str) -> None:
    settings = get_settings()
    dims = settings.embedding_dimensions
    threshold = settings.match_threshold

    classes = CLASSES[:n_classes]
    name_pool = cycle(
        f"{f} {l}" for f in random.sample(FIRST, len(FIRST))
        for l in random.sample(LAST, len(LAST))
    )

    # ── create students + embeddings ──────────────────────────────────────────
    t0 = time.perf_counter()
    students: list[tuple[Student, list[float]]] = []

    async with SessionLocal() as s:
        for i in range(n_students):
            class_id = classes[i % n_classes]
            student_id = f"{class_id}-{i+1:04d}"
            name = next(name_pool)

            # skip if already exists
            existing = await s.scalar(select(Student).where(Student.student_id == student_id))
            if existing:
                # fetch embedding for event generation
                emb_row = await s.scalar(
                    select(StudentEmbedding).where(StudentEmbedding.student_id == existing.id)
                )
                if emb_row:
                    students.append((existing, emb_row.embedding))
                continue

            st = Student(
                student_id=student_id,
                name=name,
                class_id=class_id,
                status=StudentStatus.active,
            )
            s.add(st)
            await s.flush()  # get st.id

            # 3–5 enrollment embeddings per student (slight variations)
            base_emb = make_embedding(st.id, dims)
            for _ in range(random.randint(3, 5)):
                s.add(StudentEmbedding(
                    student_id=st.id,
                    embedding=perturb(base_emb, 0.05),
                    model_version=model_version,
                    active=True,
                ))
            students.append((st, base_emb))

        await s.commit()

    t_students = time.perf_counter() - t0
    print(f"students: {len(students)} created/loaded in {t_students:.2f}s")

    # ── generate attendance events ────────────────────────────────────────────
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    days = [now - timedelta(days=d) for d in range(n_days - 1, -1, -1)]

    total_events = 0
    t_events = time.perf_counter()
    event_status_counts: dict[str, int] = {}

    # build a lookup: student.id → base_embedding for recognized events
    emb_by_id = {st.id: emb for st, emb in students}

    BATCH = 200
    event_rows: list[tuple] = []   # (job, event) pairs to flush in batches

    async with SessionLocal() as s:
        for day in days:
            for session_base in session_times(day):
                # random subset attends each session (70–95% attendance)
                attendees = random.sample(students, k=int(len(students) * random.uniform(0.70, 0.95)))

                for st, base_emb in attendees:
                    status = random.choices(STATUSES, weights=WEIGHTS, k=1)[0]
                    captured_at = jitter(session_base.replace(tzinfo=timezone.utc))
                    camera_id = random.choice(CAMERAS)

                    job = ImageJob(
                        image_path=f"seed:synthetic:{uuid.uuid4()}",
                        camera_id=camera_id,
                        class_id=st.class_id,
                        captured_at=captured_at,
                        status=JobStatus.complete,
                        processed_at=captured_at,
                    )
                    s.add(job)
                    await s.flush()

                    if status == AttendanceStatus.recognized:
                        # query embedding = slight perturbation of their own
                        confidence = random.uniform(threshold, min(threshold + 0.35, 1.0))
                        event = AttendanceEvent(
                            student_id=st.id,
                            candidate_student_id=st.id,
                            job_id=job.id,
                            camera_id=camera_id,
                            class_id=st.class_id,
                            captured_at=captured_at,
                            confidence=confidence,
                            status=AttendanceStatus.recognized,
                        )
                    elif status == AttendanceStatus.low_confidence:
                        # picked someone as candidate but below threshold
                        candidate = random.choice(students)[0]
                        confidence = random.uniform(threshold * 0.5, threshold - 0.01)
                        event = AttendanceEvent(
                            student_id=None,
                            candidate_student_id=candidate.id,
                            job_id=job.id,
                            camera_id=camera_id,
                            class_id=st.class_id,
                            captured_at=captured_at,
                            confidence=confidence,
                            status=AttendanceStatus.low_confidence,
                        )
                    elif status == AttendanceStatus.unknown:
                        event = AttendanceEvent(
                            student_id=None,
                            candidate_student_id=None,
                            job_id=job.id,
                            camera_id=camera_id,
                            class_id=st.class_id,
                            captured_at=captured_at,
                            confidence=None,
                            status=AttendanceStatus.unknown,
                        )
                    else:
                        event = AttendanceEvent(
                            student_id=None,
                            candidate_student_id=None,
                            job_id=job.id,
                            camera_id=camera_id,
                            class_id=st.class_id,
                            captured_at=captured_at,
                            confidence=None,
                            status=status,
                        )

                    s.add(event)
                    total_events += 1
                    event_status_counts[status.value] = event_status_counts.get(status.value, 0) + 1

                    if total_events % BATCH == 0:
                        await s.commit()
                        print(f"  ... {total_events} events written", end="\r")

        await s.commit()

    t_events = time.perf_counter() - t_events
    print(f"\nevents:   {total_events} written in {t_events:.2f}s  ({total_events/t_events:.0f} events/s)")

    print("\nstatus breakdown:")
    for k, v in sorted(event_status_counts.items(), key=lambda x: -x[1]):
        bar = "█" * (v * 30 // total_events)
        print(f"  {k:<30} {v:>6}  {bar}")

    print(f"\ntotal time: {time.perf_counter() - t0:.2f}s")
    print(f"\ndashboard → http://localhost:8501")
    print(f"api       → http://localhost:8000")


async def main(args: argparse.Namespace) -> None:
    await create_tables_for_local_dev()
    if args.reset:
        await reset_seed_data()
    settings = get_settings()
    await seed(
        n_students=args.students,
        n_classes=min(args.classes, len(CLASSES)),
        n_days=args.days,
        model_version=settings.worker_model_version,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed DB with synthetic students and attendance.")
    p.add_argument("--students",  type=int, default=100, help="Number of students to create")
    p.add_argument("--classes",   type=int, default=5,   help="Number of class groups")
    p.add_argument("--days",      type=int, default=7,   help="Days of history to generate")
    p.add_argument("--reset",     action="store_true",   help="Wipe existing data first")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
