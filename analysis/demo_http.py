"""Full-stack HTTP recognition demo against the live FastAPI server.

Registers a few throwaway students (DEMO-xxx) via the real REST endpoints,
waits for the batch embedder, then hits /checkin/recognize on held-out images
and an impostor. Prints the server's JSON verdicts.

The caller is responsible for DB snapshot/restore and folder cleanup — this
script only drives the HTTP API.

    python -m analysis.demo_http --api http://localhost:8000 --people 5
"""
from __future__ import annotations

import argparse
import io
import sys
import time

import requests
from sklearn.datasets import fetch_olivetti_faces

from tests.stress.image_seed import face_to_frame, to_jpeg_bytes, augment_params

ENROLL = 5
TEST = 2
IMPOSTOR_PID = 39


def wait_healthy(api, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            h = requests.get(f"{api}/health", timeout=3).json()
            if h.get("face_model_loaded"):
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def register(api, sid, name, faces, pid):
    files = []
    for idx, face in enumerate(faces):
        b, ang = augment_params(pid * 100 + idx)
        frame = face_to_frame(face, brightness=b, rotate_deg=ang)
        files.append(("images", (f"e{idx}.jpg", io.BytesIO(to_jpeg_bytes(frame)), "image/jpeg")))
    r = requests.post(f"{api}/students/register",
                      data={"student_id": sid, "name": name, "class_id": "DEMO-101"},
                      files=files, timeout=30)
    return r.status_code, r.json() if r.headers.get("content-type", "").startswith("application/json") else {}


def recognize(api, face, brightness=1.0, rotate=0.0):
    frame = face_to_frame(face, brightness=brightness, rotate_deg=rotate)
    files = {"image": ("q.jpg", io.BytesIO(to_jpeg_bytes(frame)), "image/jpeg")}
    r = requests.post(f"{api}/checkin/recognize", data={"camera_id": "demo-kiosk"},
                      files=files, timeout=20)
    return r.json()


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--people", type=int, default=5)
    args = ap.parse_args()
    api = args.api.rstrip("/")

    print("=" * 66)
    print("  LIVE HTTP RECOGNITION TEST  (FastAPI + SQLite + YuNet/ArcFace)")
    print("=" * 66)
    if not wait_healthy(api):
        print("  ERROR: server not healthy / model not loaded"); sys.exit(1)
    print(f"  server healthy at {api}\n")

    data = fetch_olivetti_faces(shuffle=False)
    people = list(range(args.people))
    ids = [f"DEMO-{p+1:03d}" for p in people]

    # 1) Register
    print(f"[1] POST /students/register  ({len(people)} people x {ENROLL} imgs)")
    for p, sid in zip(people, ids):
        imgs = data.images[data.target == p][:ENROLL]
        code, body = register(api, sid, f"Demo Person {p+1}", imgs, p)
        print(f"    {sid}: HTTP {code}  saved={body.get('images_saved')}")

    # 2) Trigger embedding + poll status
    print("\n[2] POST /students/{id}/process  then GET /students/{id}/status")
    requests.post(f"{api}/students/{ids[0]}/process", timeout=10)
    active = []
    for sid in ids:
        st, n = "pending", 0
        for _ in range(80):
            d = requests.get(f"{api}/students/{sid}/status", timeout=5).json()
            st, n = d["status"], d.get("active_embeddings", 0)
            if st in ("active", "embedding_failed"):
                break
            time.sleep(1.0)
        print(f"    {sid}: {st}  embeddings={n}")
        if st == "active":
            active.append((sid, [p for p, s in zip(people, ids) if s == sid][0]))

    # 3) Recognise held-out genuine images
    print(f"\n[3] POST /checkin/recognize  ({len(active)*TEST} held-out genuine images)\n")
    print(f"    {'true':<10}{'server status':<15}{'matched':<10}{'conf':>7}  result")
    print("    " + "-" * 55)
    correct = total = 0
    for sid, pid in active:
        held = data.images[data.target == pid][ENROLL:ENROLL + TEST]
        for k, face in enumerate(held):
            b, ang = augment_params(pid * 100 + ENROLL + k)
            res = recognize(api, face, b, ang)
            total += 1
            matched = res.get("student_ext_id")
            ok = res.get("status") == "recognized" and matched == sid
            correct += ok
            print(f"    {sid:<10}{res.get('status',''):<15}{str(matched):<10}"
                  f"{str(res.get('confidence','-')):>7}  {'OK' if ok else res.get('status')}")

    # 4) Impostor (unenrolled)
    print(f"\n[4] Impostor — Olivetti person {IMPOSTOR_PID} was never registered:")
    res = recognize(api, data.images[data.target == IMPOSTOR_PID][0])
    print(f"    status={res.get('status')}  matched={res.get('student_ext_id')}  "
          f"conf={res.get('confidence')}  threshold={res.get('threshold_used')}")

    print("\n" + "=" * 66)
    print(f"  Genuine recognition: {correct}/{total} correct "
          f"({(correct/total*100 if total else 0):.1f}%)")
    print(f"  Impostor verdict   : {res.get('status')} "
          f"(recognised=false-accept, unknown/low_confidence=correct-reject)")
    print("=" * 66)


if __name__ == "__main__":
    main()
