"""Webcam helper to register and recognize a face end-to-end.

Examples:

    # Enroll a student from the webcam (captures 5 good frames)
    python -m tools.face_cli register --student-id STU-001 --name "Alex Johnson"

    # Or enroll from existing image files
    python -m tools.face_cli register --student-id STU-001 --name "Alex Johnson" \
        --image a.jpg --image b.jpg --image c.jpg

    # Test recognition from the webcam
    python -m tools.face_cli recognize

    # Test recognition from a file
    python -m tools.face_cli recognize --image selfie.jpg

Needs the OpenCV Zoo models (run `python -m worker.download_models` first).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from uuid import UUID, uuid4

import cv2
import numpy as np
from sqlalchemy import select

from core.config import get_settings
from core.database import SessionLocal
from core.init_db import create_tables_for_local_dev
from core.models import Student, StudentEmbedding, StudentStatus
from worker.model_adapter import FaceEmbeddingModel
from worker.registration_updater import process_pending_registrations
from worker.runner import match_student

REGISTRATION_ROOT = Path("work/registrations")


def open_camera(camera_index: int) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    capture = cv2.VideoCapture(camera_index, backend)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open webcam at index {camera_index}. "
            "Close other apps using the camera, try --camera 1, or pass --image."
        )
    return capture


def capture_face_frames(
    model: FaceEmbeddingModel,
    shots: int,
    camera_index: int,
    delay_seconds: float,
    max_seconds: float,
) -> list[np.ndarray]:
    """Grab `shots` frames that each contain exactly one detectable face."""
    capture = open_camera(camera_index)
    try:
        for _ in range(10):  # let the sensor auto-expose
            capture.read()
        if delay_seconds > 0:
            print(f"Look at the camera... capturing in {delay_seconds:.0f}s")
            time.sleep(delay_seconds)

        frames: list[np.ndarray] = []
        deadline = time.time() + max_seconds
        while len(frames) < shots and time.time() < deadline:
            ok, frame = capture.read()
            if not ok:
                continue
            _, face_count = model.embed_array(frame)
            if face_count == 1:
                frames.append(frame.copy())
                print(f"  captured {len(frames)}/{shots}")
                time.sleep(0.4)
            elif face_count > 1:
                print("  multiple faces in frame, hold still and stay solo...")
                time.sleep(0.2)
            else:
                time.sleep(0.1)
        return frames
    finally:
        capture.release()


def load_image_frames(paths: list[Path]) -> list[np.ndarray]:
    frames = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"  skip unreadable image: {path}")
            continue
        frames.append(image)
    return frames


async def register(args: argparse.Namespace) -> int:
    await create_tables_for_local_dev()
    settings = get_settings()
    model = FaceEmbeddingModel.from_settings(settings)

    if args.image:
        frames = load_image_frames([Path(p) for p in args.image])
    else:
        frames = capture_face_frames(
            model, args.shots, args.camera, args.delay, args.max_seconds
        )

    if len(frames) < 3:
        print(f"Only collected {len(frames)} usable face image(s); need at least 3.")
        return 1

    target_dir = REGISTRATION_ROOT / args.student_id
    target_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        cv2.imwrite(str(target_dir / f"{uuid4()}.jpg"), frame)
    print(f"Saved {len(frames)} image(s) to {target_dir}")

    async with SessionLocal() as session:
        student = await session.scalar(
            select(Student).where(Student.student_id == args.student_id)
        )
        if student is None:
            session.add(
                Student(
                    student_id=args.student_id,
                    name=args.name,
                    class_id=args.class_id,
                    status=StudentStatus.pending_embedding,
                )
            )
        else:
            student.name = args.name
            student.status = StudentStatus.pending_embedding
        await session.commit()

    processed = await process_pending_registrations()
    print(f"Embedded registrations for {processed} pending student(s).")

    async with SessionLocal() as session:
        student = await session.scalar(
            select(Student).where(Student.student_id == args.student_id)
        )
        active_count = len(
            (
                await session.execute(
                    select(StudentEmbedding.id).where(
                        StudentEmbedding.student_id == student.id,
                        StudentEmbedding.active.is_(True),
                    )
                )
            ).all()
        )
    print(f"Student '{args.student_id}' status={student.status.value}, active_embeddings={active_count}")
    return 0 if student.status == StudentStatus.active else 1


async def resolve_label(student_id: UUID | None) -> str | None:
    if student_id is None:
        return None
    async with SessionLocal() as session:
        student = await session.get(Student, student_id)
        return f"{student.student_id} ({student.name})" if student else str(student_id)


async def recognize(args: argparse.Namespace) -> int:
    await create_tables_for_local_dev()
    settings = get_settings()
    model = FaceEmbeddingModel.from_settings(settings)

    if args.image:
        frames = load_image_frames([Path(args.image)])
    else:
        frames = capture_face_frames(model, 1, args.camera, args.delay, args.max_seconds)

    if not frames:
        print("No usable frame captured.")
        return 1

    embedding, face_count = model.embed_array(frames[0])
    if face_count == 0 or embedding is None:
        print("status=no_face_detected")
        return 1
    if face_count > 1:
        print(f"status=multiple_faces_detected (faces={face_count})")
        return 1

    candidate_id, best_score = await match_student(embedding)
    threshold = settings.match_threshold
    label = await resolve_label(candidate_id)

    if candidate_id is None:
        status = "unknown (no enrolled students)"
    elif best_score >= threshold:
        status = f"recognized -> {label}"
    else:
        status = f"low_confidence (closest: {label})"

    print(f"status={status}")
    print(f"cosine_score={best_score:.4f}  threshold={threshold}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register and recognize faces from the webcam.")
    sub = parser.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("register", help="Enroll a student from webcam or images.")
    reg.add_argument("--student-id", required=True, help="Unique ID for this person, e.g. stu-001")
    reg.add_argument("--name", required=True, help="Full name")
    reg.add_argument("--class-id", default=None)
    reg.add_argument("--shots", type=int, default=5)
    reg.add_argument("--image", action="append", help="Use image file(s) instead of webcam.")

    rec = sub.add_parser("recognize", help="Identify the face in a frame.")
    rec.add_argument("--image", help="Use an image file instead of webcam.")

    for sp in (reg, rec):
        sp.add_argument("--camera", type=int, default=0)
        sp.add_argument("--delay", type=float, default=3.0, help="Seconds before capture starts.")
        sp.add_argument("--max-seconds", type=float, default=25.0, help="Capture window timeout.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    handler = register if args.command == "register" else recognize
    raise SystemExit(asyncio.run(handler(args)))


if __name__ == "__main__":
    main()
