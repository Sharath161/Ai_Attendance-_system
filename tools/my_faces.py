"""Register and recognise YOUR OWN people from image files.

Folder layout — one subfolder per person, photos inside:

    work/my_people/
        Sharath/         img1.jpg  img2.jpg  img3.jpg ...
        Alex Johnson/    a.png     b.png ...
        Priya/           1.jpg     2.jpg ...

Then:

    # 1. enrol everyone in work/my_people/  (3-5 clear, front-facing photos each)
    python -m tools.my_faces register --dir work/my_people

    # 2. identify a new photo
    python -m tools.my_faces recognize --image path/to/selfie.jpg

    # 3. batch-identify every image in a folder
    python -m tools.my_faces recognize --dir path/to/test_photos

    # helpers
    python -m tools.my_faces list                 # who is enrolled
    python -m tools.my_faces remove --id Sharath  # remove one person
    python -m tools.my_faces remove --all-mine    # remove everyone added by this tool

Tips: use sharp, well-lit, single-face photos. More angles = better accuracy.
Models required once:  python -m worker.download_models
"""
from __future__ import annotations

import argparse
import asyncio
import re
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from sqlalchemy import select, delete

from core.config import get_settings
from core.database import SessionLocal
from core.init_db import create_tables_for_local_dev
from core.models import Student, StudentEmbedding, StudentStatus
from worker.model_adapter import FaceEmbeddingModel
from worker.registration_updater import process_pending_registrations

REG_ROOT = Path("work/registrations")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# student_ids created by this tool carry a marker so we can list/remove them safely
MINE_CLASS = "my-people"


def slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", name.strip()).strip("-")
    return s or "person"


def _reconfig_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ── register ──────────────────────────────────────────────────────────────────
async def register(args) -> int:
    _reconfig_stdout()
    await create_tables_for_local_dev()
    settings = get_settings()
    model = FaceEmbeddingModel.from_settings(settings)

    root = Path(args.dir)
    if not root.exists():
        print(f"Folder not found: {root}")
        return 1
    people = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if not people:
        print(f"No person-subfolders inside {root}. Create one folder per person.")
        return 1

    print(f"Found {len(people)} person folder(s) in {root}\n")
    to_process: list[tuple[str, str]] = []

    for d in people:
        name = d.name
        sid = slugify(name)
        imgs = [p for p in sorted(d.glob("*")) if p.suffix.lower() in IMG_EXT]
        print(f"• {name}  (id={sid}) — {len(imgs)} file(s)")
        if not imgs:
            print("    no images, skipped"); continue

        usable: list[np.ndarray] = []
        for p in imgs:
            im = cv2.imread(str(p))
            if im is None:
                print(f"    {p.name}: unreadable, skipped"); continue
            _, fc = model.embed_array(im)
            if fc == 1:
                usable.append(im)
            elif fc > 1:
                print(f"    {p.name}: {fc} faces (need exactly 1), skipped")
            else:
                print(f"    {p.name}: no face detected, skipped")
        if not usable:
            print("    no usable single-face photos, skipped"); continue

        # fresh registration folder (idempotent re-enrol)
        target = REG_ROOT / sid
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        for im in usable:
            cv2.imwrite(str(target / f"{uuid4()}.jpg"), im)
        print(f"    saved {len(usable)} usable photo(s)")

        # (re)create the student row, clearing any previous embeddings
        async with SessionLocal() as session:
            student = await session.scalar(select(Student).where(Student.student_id == sid))
            if student is not None:
                await session.execute(
                    delete(StudentEmbedding).where(StudentEmbedding.student_id == student.id)
                )
                student.name = name
                student.class_id = MINE_CLASS
                student.status = StudentStatus.pending_embedding
            else:
                session.add(Student(student_id=sid, name=name,
                                    class_id=MINE_CLASS,
                                    status=StudentStatus.pending_embedding))
            await session.commit()
        to_process.append((sid, name))

    if not to_process:
        print("\nNothing to enrol.")
        return 1

    print("\nGenerating embeddings ...")
    await process_pending_registrations(limit=max(25, len(to_process)))

    print("\nEnrolment result:")
    async with SessionLocal() as session:
        for sid, name in to_process:
            student = await session.scalar(select(Student).where(Student.student_id == sid))
            n = (await session.execute(
                select(StudentEmbedding.id).where(
                    StudentEmbedding.student_id == student.id,
                    StudentEmbedding.active.is_(True)))).all()
            flag = "OK" if student.status == StudentStatus.active else "FAILED"
            print(f"  [{flag}] {name:<22} id={sid:<18} embeddings={len(n)}")
    print("\nDone. Now recognise a photo:  python -m tools.my_faces recognize --image <file>")
    return 0


# ── recognise ─────────────────────────────────────────────────────────────────
async def _load_gallery(session):
    """Return {student_id_ext: (name, [embeddings])} for all active embeddings."""
    rows = (await session.execute(
        select(Student.student_id, Student.name, StudentEmbedding.embedding)
        .join(StudentEmbedding, StudentEmbedding.student_id == Student.id)
        .where(StudentEmbedding.active.is_(True)))).all()
    gal: dict[str, tuple[str, list]] = {}
    for ext_id, name, emb in rows:
        gal.setdefault(ext_id, (name, []))[1].append(np.asarray(emb, dtype=np.float32))
    return gal


def _identify(model, frame, gallery, threshold, topk=3):
    emb, fc, _ = model.detect_and_embed(frame)
    if fc == 0 or emb is None:
        return {"status": "no_face"}
    if fc > 1:
        return {"status": "multiple_faces", "faces": fc}
    scores = []
    for ext_id, (name, embs) in gallery.items():
        best = max(float(np.dot(emb, e)) for e in embs)   # best of that person's shots
        scores.append((ext_id, name, best))
    scores.sort(key=lambda x: -x[2])
    top = scores[:topk]
    if not top:
        return {"status": "unknown", "top": []}
    best_id, best_name, best_score = top[0]
    status = "recognised" if best_score >= threshold else "low_confidence"
    return {"status": status, "best_id": best_id, "best_name": best_name,
            "best_score": best_score, "top": top}


async def recognize(args) -> int:
    _reconfig_stdout()
    await create_tables_for_local_dev()
    settings = get_settings()
    model = FaceEmbeddingModel.from_settings(settings)
    thr = settings.match_threshold

    async with SessionLocal() as session:
        gallery = await _load_gallery(session)
    if not gallery:
        print("No one is enrolled yet. Run:  python -m tools.my_faces register --dir work/my_people")
        return 1
    print(f"Gallery: {len(gallery)} enrolled — {', '.join(sorted(gallery))}\n")

    targets: list[Path] = []
    if args.image:
        targets = [Path(args.image)]
    elif args.dir:
        targets = [p for p in sorted(Path(args.dir).glob("*")) if p.suffix.lower() in IMG_EXT]
    if not targets:
        print("Provide --image <file> or --dir <folder>."); return 1

    for path in targets:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"{path.name}: unreadable"); continue
        r = _identify(model, frame, gallery, thr)
        if r["status"] in ("no_face", "multiple_faces"):
            print(f"{path.name:<28} -> {r['status']}"); continue
        verdict = (f"{r['best_name']} (id={r['best_id']})" if r["status"] == "recognised"
                   else f"closest {r['best_name']} but below threshold")
        print(f"{path.name:<28} -> {r['status'].upper():<14} {verdict}   score={r['best_score']:.3f} (thr={thr})")
        if args.verbose and r.get("top"):
            for ext_id, name, sc in r["top"]:
                print(f"      {sc:6.3f}  {name} ({ext_id})")
    return 0


# ── list / remove ─────────────────────────────────────────────────────────────
async def list_people(args) -> int:
    _reconfig_stdout()
    await create_tables_for_local_dev()
    async with SessionLocal() as session:
        rows = (await session.execute(
            select(Student.student_id, Student.name, Student.class_id, Student.status))).all()
        print(f"{'id':<20}{'name':<24}{'class':<12}{'status':<12}{'embeddings'}")
        print("-" * 78)
        for ext_id, name, cls, status in rows:
            student = await session.scalar(select(Student).where(Student.student_id == ext_id))
            n = (await session.execute(select(StudentEmbedding.id).where(
                StudentEmbedding.student_id == student.id,
                StudentEmbedding.active.is_(True)))).all()
            print(f"{ext_id:<20}{name:<24}{str(cls):<12}{status.value:<12}{len(n)}")
    return 0


async def remove(args) -> int:
    _reconfig_stdout()
    await create_tables_for_local_dev()
    async with SessionLocal() as session:
        q = select(Student)
        if args.all_mine:
            q = q.where(Student.class_id == MINE_CLASS)
        elif args.id:
            q = q.where(Student.student_id == args.id)
        else:
            print("Specify --id <student_id> or --all-mine"); return 1
        students = (await session.execute(q)).scalars().all()
        if not students:
            print("Nothing matched."); return 0
        for s in students:
            await session.execute(delete(StudentEmbedding).where(StudentEmbedding.student_id == s.id))
            folder = REG_ROOT / s.student_id
            if folder.exists():
                shutil.rmtree(folder, ignore_errors=True)
            print(f"removed {s.student_id} ({s.name})")
            await session.delete(s)
        await session.commit()
    return 0


def parse_args():
    p = argparse.ArgumentParser(description="Register & recognise your own people from image files.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register", help="Enrol everyone in a folder (one subfolder per person).")
    r.add_argument("--dir", default="work/my_people")

    g = sub.add_parser("recognize", help="Identify a photo (or a folder of photos).")
    g.add_argument("--image"); g.add_argument("--dir"); g.add_argument("--verbose", action="store_true")

    sub.add_parser("list", help="Show enrolled people.")

    d = sub.add_parser("remove", help="Remove enrolled people.")
    d.add_argument("--id"); d.add_argument("--all-mine", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    fn = {"register": register, "recognize": recognize,
          "list": list_people, "remove": remove}[args.cmd]
    raise SystemExit(asyncio.run(fn(args)))


if __name__ == "__main__":
    main()
