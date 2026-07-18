from __future__ import annotations

import argparse
import asyncio
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from core.config import get_settings
from core.database import SessionLocal
from core.models import ImageJob
from worker.runner import process_next_job
from worker.model_adapter import FaceEmbeddingModel


async def seed_jobs(sample_image: Path, jobs: int, spool_dir: Path) -> None:
    spool_dir.mkdir(parents=True, exist_ok=True)
    async with SessionLocal() as session:
        for index in range(jobs):
            target = spool_dir / f"throughput-{index}.jpg"
            shutil.copyfile(sample_image, target)
            session.add(
                ImageJob(
                    image_path=str(target),
                    camera_id="throughput-test",
                    class_id="validation",
                    captured_at=datetime.now(timezone.utc),
                )
            )
        await session.commit()


async def measure(sample_image: Path, jobs: int, spool_dir: Path, model_version: str) -> None:
    await seed_jobs(sample_image, jobs, spool_dir)
    settings = get_settings()
    settings.worker_model_version = model_version
    model = FaceEmbeddingModel.from_settings(settings)

    start = time.perf_counter()
    processed = 0
    for _ in range(jobs):
        if await process_next_job(model):
            processed += 1
    elapsed = time.perf_counter() - start

    per_image_ms = (elapsed / processed) * 1000 if processed else 0
    print(f"processed={processed}")
    print(f"elapsed_seconds={elapsed:.4f}")
    print(f"milliseconds_per_image={per_image_ms:.2f}")
    print(f"estimated_12000_image_clearance_minutes={(12000 * per_image_ms / 1000) / 60:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure sequential worker throughput.")
    parser.add_argument("--sample-image", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=100)
    parser.add_argument("--spool-dir", type=Path, default=Path("work/throughput-spool"))
    parser.add_argument("--model-version", default="mobilefacenet-placeholder-v1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(measure(args.sample_image, args.jobs, args.spool_dir, args.model_version))
