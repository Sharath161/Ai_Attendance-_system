import asyncio
import json
from pathlib import Path

from sqlalchemy import select

from core.config import get_settings
from core.database import SessionLocal
from core.init_db import create_tables_for_local_dev
from core.models import Student, StudentEmbedding, StudentStatus
from worker.model_adapter import FaceEmbeddingModel


async def process_pending_registrations(limit: int = 25) -> int:
    await create_tables_for_local_dev()
    settings = get_settings()
    model = FaceEmbeddingModel.from_settings(settings)
    processed = 0

    async with SessionLocal() as session:
        result = await session.execute(
            select(Student)
            .where(Student.status == StudentStatus.pending_embedding)
            .order_by(Student.created_at)
            .limit(limit)
        )
        students = result.scalars().all()

        for student in students:
            # Absolute path — works regardless of cwd
            registration_dir = settings.spool_dir.parent / "registrations" / student.student_id

            if not registration_dir.exists():
                print(f"[reg] registration dir missing for {student.student_id}: {registration_dir}")
                student.status = StudentStatus.embedding_failed
                processed += 1
                continue

            # Read optional demographic metadata written by the register endpoint
            demographic_group: str | None = None
            meta_path = registration_dir / "metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    demographic_group = meta.get("demographic_group")
                except Exception:
                    pass

            valid_embeddings = 0
            image_files = sorted(
                p for p in registration_dir.glob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )
            print(f"[reg] processing {student.student_id}: {len(image_files)} images found")

            for image_path in image_files:
                embedding = model.embed_image(image_path)
                if embedding is None:
                    print(f"[reg]   no face detected in {image_path.name}")
                    continue

                session.add(
                    StudentEmbedding(
                        student_id=student.id,
                        embedding=embedding.tolist(),
                        model_version=model.model_version,
                        image_quality_score=None,
                        active=True,
                        demographic_group=demographic_group,
                    )
                )
                valid_embeddings += 1
                print(f"[reg]   embedded {image_path.name} ({valid_embeddings} so far)")

            # Need at least 1 valid embedding for the demo to work
            student.status = (
                StudentStatus.active
                if valid_embeddings >= 1
                else StudentStatus.embedding_failed
            )
            print(f"[reg] {student.student_id} -> {student.status.value} ({valid_embeddings} embeddings)")
            processed += 1

        await session.commit()

    return processed


if __name__ == "__main__":
    count = asyncio.run(process_pending_registrations())
    print(f"processed_pending_registrations={count}")
