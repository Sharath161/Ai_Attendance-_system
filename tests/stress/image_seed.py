"""
Stress test using the Olivetti faces dataset.

40 real people × 10 photos each → 5 classes of 8 students.
  • Images 0–6  used for enrollment  (7 per person)
  • Images 7–9  held out for recognition testing (3 per person)

Steps
-----
1. Prepare face images (64×64 grayscale → 640×480 BGR JPEG in memory)
2. Register every student via POST /students/register
3. Poll GET /students/{id}/status until active or failed
4. Run POST /checkin/recognize on hold-out images
5. Report per-class accuracy, overall F1, and throughput

Usage
-----
    python -m tests.stress.image_seed
    python -m tests.stress.image_seed --api http://localhost:8000 --classes 5
    python -m tests.stress.image_seed --reset   # wipe DB first, then seed
"""
from __future__ import annotations

import argparse
import io
import random
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import requests
from sklearn.datasets import fetch_olivetti_faces

ENROLL_PER_PERSON = 7   # images used for registration
TEST_PER_PERSON   = 3   # held-out images for recognition check

CLASS_NAMES = ["CS-101", "CS-102", "MATH-201", "PHY-301", "ENG-101",
               "BIO-110", "CHEM-201", "ECON-101"]

# diverse generic names — no personal names
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
    "White", "Harris", "Martin", "Thompson", "Lee", "Walker",
    "Patel", "Khan", "Singh", "Nguyen", "Kim", "Tanaka", "Okafor", "Santos",
    "Rossi", "Müller", "Costa", "Ferreira", "Yamamoto", "Nakamura", "Petrov",
    "Ivanova", "Dubois", "Lefebvre", "Osei", "Mensah",
]


# ── image helpers ─────────────────────────────────────────────────────────────

def face_to_frame(
    face: np.ndarray,
    face_size: int = 280,
    canvas_w: int = 640,
    canvas_h: int = 480,
    brightness: float = 1.0,
    rotate_deg: float = 0.0,
) -> np.ndarray:
    """Convert a 64×64 float face → 640×480 uint8 BGR ready for YuNet."""
    face_u8 = np.clip(face * 255 * brightness, 0, 255).astype(np.uint8)
    face_big = cv2.resize(face_u8, (face_size, face_size), interpolation=cv2.INTER_LANCZOS4)
    face_bgr = cv2.cvtColor(face_big, cv2.COLOR_GRAY2BGR)

    if rotate_deg != 0.0:
        cx, cy = face_size // 2, face_size // 2
        M = cv2.getRotationMatrix2D((cx, cy), rotate_deg, 1.0)
        face_bgr = cv2.warpAffine(face_bgr, M, (face_size, face_size), borderValue=(140, 140, 140))

    canvas = np.full((canvas_h, canvas_w, 3), 140, dtype=np.uint8)
    y0 = (canvas_h - face_size) // 2
    x0 = (canvas_w - face_size) // 2
    canvas[y0:y0+face_size, x0:x0+face_size] = face_bgr
    return canvas


def to_jpeg_bytes(frame: np.ndarray, quality: int = 90) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def augment_params(seed: int) -> tuple[float, float]:
    """Return (brightness, rotation) for slight per-image variation."""
    rng = random.Random(seed)
    return (
        rng.uniform(0.85, 1.15),   # brightness ±15%
        rng.uniform(-6.0, 6.0),    # rotation ±6°
    )


# ── student record ────────────────────────────────────────────────────────────

@dataclass
class FakeStudent:
    person_id: int          # Olivetti person index 0-39
    student_id: str
    name: str
    class_id: str
    enroll_imgs: list[np.ndarray]
    test_imgs:   list[np.ndarray]
    api_status: str = "pending"
    active_embeddings: int = 0
    test_results: list[dict] = field(default_factory=list)


# ── API helpers ───────────────────────────────────────────────────────────────

def register(api: str, student: FakeStudent, timeout: int = 30) -> bool:
    files = []
    for idx, face in enumerate(student.enroll_imgs):
        brightness, angle = augment_params(student.person_id * 100 + idx)
        frame = face_to_frame(face, brightness=brightness, rotate_deg=angle)
        files.append(("images", (f"enroll_{idx}.jpg", io.BytesIO(to_jpeg_bytes(frame)), "image/jpeg")))

    data = {
        "student_id": student.student_id,
        "name": student.name,
        "class_id": student.class_id,
    }
    try:
        r = requests.post(f"{api}/students/register", data=data, files=files, timeout=timeout)
        if r.status_code == 409:
            return True   # already registered
        return r.status_code == 202
    except Exception as e:
        print(f"  register error [{student.student_id}]: {e}")
        return False


def trigger_process(api: str, student_id: str) -> None:
    try:
        requests.post(f"{api}/students/{student_id}/process", timeout=10)
    except Exception:
        pass


def poll_status(api: str, student_id: str, max_wait: float = 60.0) -> tuple[str, int]:
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = requests.get(f"{api}/students/{student_id}/status", timeout=5)
            d = r.json()
            if d["status"] in ("active", "embedding_failed"):
                return d["status"], d.get("active_embeddings", 0)
        except Exception:
            pass
        time.sleep(1.5)
    return "timeout", 0


def recognize(api: str, face: np.ndarray, camera_id: str = "stress-test") -> dict:
    frame = face_to_frame(face)
    form = {"camera_id": camera_id}
    files = {"image": ("test.jpg", io.BytesIO(to_jpeg_bytes(frame)), "image/jpeg")}
    try:
        r = requests.post(f"{api}/checkin/recognize", data=form, files=files, timeout=15)
        return r.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def reset_db(api: str) -> None:
    """Call the seed_data reset via direct DB (runs locally)."""
    import asyncio
    from tests.stress.seed_data import reset_seed_data
    asyncio.run(reset_seed_data())
    print("db reset complete")


# ── reporting ─────────────────────────────────────────────────────────────────

def report(students: list[FakeStudent]) -> None:
    print("\n" + "═" * 60)
    print("RESULTS")
    print("═" * 60)

    per_class: dict[str, dict] = {}

    tp = fp = fn = tn = 0
    for s in students:
        cls = s.class_id
        if cls not in per_class:
            per_class[cls] = {"correct": 0, "total": 0, "status_counts": {}}
        for r in s.test_results:
            per_class[cls]["total"] += 1
            status = r.get("status", "error")
            per_class[cls]["status_counts"][status] = per_class[cls]["status_counts"].get(status, 0) + 1

            if status == "recognized" and r.get("student_ext_id") == s.student_id:
                per_class[cls]["correct"] += 1
                tp += 1
            elif status == "recognized":
                fp += 1   # recognized but wrong person
            elif status in ("unknown", "low_confidence", "no_face", "error"):
                fn += 1   # should have been recognized

    print(f"\n{'Class':<12} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 44)
    for cls, d in sorted(per_class.items()):
        acc = d["correct"] / d["total"] * 100 if d["total"] else 0
        bar = "█" * int(acc / 5)
        print(f"{cls:<12} {d['correct']:>8} {d['total']:>8}   {acc:>5.1f}%  {bar}")

    total_tests = sum(d["total"] for d in per_class.values())
    total_correct = sum(d["correct"] for d in per_class.values())
    overall_acc = total_correct / total_tests * 100 if total_tests else 0

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("-" * 44)
    print(f"{'TOTAL':<12} {total_correct:>8} {total_tests:>8}   {overall_acc:>5.1f}%")
    print(f"\nPrecision: {precision:.3f}   Recall: {recall:.3f}   F1: {f1:.3f}")

    # embedding stats
    active   = sum(1 for s in students if s.api_status == "active")
    failed   = sum(1 for s in students if s.api_status == "embedding_failed")
    timedout = sum(1 for s in students if s.api_status == "timeout")
    print(f"\nEnrollment: {active} active / {failed} failed / {timedout} timed-out  (of {len(students)})")

    # status breakdown
    all_statuses: dict[str, int] = {}
    for s in students:
        for r in s.test_results:
            st = r.get("status", "error")
            all_statuses[st] = all_statuses.get(st, 0) + 1
    print("\nRecognition status breakdown:")
    for st, n in sorted(all_statuses.items(), key=lambda x: -x[1]):
        bar = "█" * (n * 30 // total_tests) if total_tests else ""
        print(f"  {st:<30} {n:>5}  {bar}")


# ── main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    api = args.api.rstrip("/")

    # sanity check
    try:
        health = requests.get(f"{api}/health", timeout=5).json()
        if not health.get("face_model_loaded"):
            print("WARNING: face model not loaded — recognition will return 503")
    except Exception:
        print(f"ERROR: API not reachable at {api}. Start it first.")
        return

    if args.reset:
        reset_db(api)

    import sys, io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("loading Olivetti faces ...")
    data = fetch_olivetti_faces(shuffle=False)
    n_people = len(set(data.target))          # 40
    n_classes = min(args.classes, len(CLASS_NAMES))

    # build name pool (deterministic)
    rng = random.Random(42)
    names = [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(n_people)]
    seen_names: set[str] = set()
    final_names: list[str] = []
    for n in names:
        while n in seen_names:
            n = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        seen_names.add(n)
        final_names.append(n)

    students: list[FakeStudent] = []
    for pid in range(n_people):
        imgs = data.images[data.target == pid]   # 10 images
        students.append(FakeStudent(
            person_id   = pid,
            student_id  = f"STU-{pid+1:03d}",
            name        = final_names[pid],
            class_id    = CLASS_NAMES[pid % n_classes],
            enroll_imgs = list(imgs[:ENROLL_PER_PERSON]),
            test_imgs   = list(imgs[ENROLL_PER_PERSON:ENROLL_PER_PERSON + TEST_PER_PERSON]),
        ))

    # ── phase 1: register ─────────────────────────────────────────────────────
    print(f"\n[1/3] Registering {len(students)} students across {n_classes} classes ...")
    t0 = time.perf_counter()
    ok = failed = 0
    for i, s in enumerate(students, 1):
        success = register(api, s)
        if success:
            ok += 1
        else:
            failed += 1
        print(f"  registered {i}/{len(students)}  ({s.class_id})  {s.name}", end="\r")
    t_reg = time.perf_counter() - t0
    print(f"\n  done: {ok} ok / {failed} failed  in {t_reg:.1f}s  ({len(students)/t_reg:.1f} students/s)")

    # trigger embedding once — the updater processes ALL pending students in one run
    print("  triggering batch embedding ...")
    trigger_process(api, students[0].student_id)

    # ── phase 2: wait for embeddings ──────────────────────────────────────────
    print(f"\n[2/3] Waiting for embeddings ...")
    t1 = time.perf_counter()
    for i, s in enumerate(students, 1):
        status, n_emb = poll_status(api, s.student_id, max_wait=120)
        s.api_status = status
        s.active_embeddings = n_emb
        icon = "OK" if status == "active" else "!!"
        print(f"  [{icon}] {i:>2}/{len(students)}  {s.student_id}  {status}  embeddings={n_emb}", end="\r")
    t_emb = time.perf_counter() - t1
    active_count = sum(1 for s in students if s.api_status == "active")
    print(f"\n  done: {active_count}/{len(students)} active in {t_emb:.1f}s")

    # ── phase 3: recognition test ─────────────────────────────────────────────
    active_students = [s for s in students if s.api_status == "active"]
    total_tests = len(active_students) * TEST_PER_PERSON
    print(f"\n[3/3] Testing recognition on {total_tests} hold-out images ...")
    t2 = time.perf_counter()
    done = 0
    for s in active_students:
        for face in s.test_imgs:
            result = recognize(api, face, camera_id=f"stress-{s.class_id}")
            s.test_results.append(result)
            done += 1
            print(f"  {done}/{total_tests}  {s.student_id}  → {result.get('status')}  "
                  f"conf={result.get('confidence', '—')}", end="\r")
    t_rec = time.perf_counter() - t2
    print(f"\n  done: {done} tests in {t_rec:.1f}s  ({done/t_rec:.1f} recognitions/s)")

    report(students)
    print(f"\ndashboard → http://localhost:8501")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stress test using Olivetti faces.")
    p.add_argument("--api",     default="http://localhost:8000")
    p.add_argument("--classes", type=int, default=5, help="Number of class groups (max 8)")
    p.add_argument("--reset",   action="store_true", help="Wipe DB before seeding")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
