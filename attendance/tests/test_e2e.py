"""End-to-end test: attendance system + real faceapi service.

Boots faceapi in a background thread on a free port, points the attendance
service at it, then drives the whole flow over HTTP:

    admin login -> create teacher/students -> course -> enrol
    -> enrol faces (real Olivetti images, via faceapi)
    -> open session -> face check-in -> duplicate blocked
    -> unknown face rejected -> manual mark -> reports
"""
from __future__ import annotations

import base64
import io
import os
import socket
import tempfile
import threading
import time
from pathlib import Path

import pytest
import uvicorn

# ── isolate both services onto throwaway databases ───────────────────────────
_TMP = Path(tempfile.mkdtemp(prefix="att_e2e_"))
os.environ["FACEAPI_DB_PATH"] = str(_TMP / "face.db")
os.environ["FACEAPI_API_KEYS"] = ""
os.environ["ATT_DB_PATH"] = str(_TMP / "att.db")
os.environ["ATT_JWT_SECRET"] = "test-secret"
os.environ["ATT_BOOTSTRAP_ADMIN_EMAIL"] = "admin@example.com"
os.environ["ATT_BOOTSTRAP_ADMIN_PASSWORD"] = "admin123"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


FACE_PORT = _free_port()
os.environ["ATT_FACEAPI_URL"] = f"http://127.0.0.1:{FACE_PORT}"

from fastapi.testclient import TestClient          # noqa: E402
from sklearn.datasets import fetch_olivetti_faces  # noqa: E402

from faceapi.api import app as face_app           # noqa: E402
from faceapi._testutil import face_to_frame, to_jpeg_bytes, augment_params  # noqa: E402
from attendance.app import app as att_app         # noqa: E402


@pytest.fixture(scope="module")
def face_server():
    """Run the real faceapi service in a background thread."""
    cfg = uvicorn.Config(face_app, host="127.0.0.1", port=FACE_PORT, log_level="warning")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    import httpx
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{FACE_PORT}/health", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.2)
    else:
        pytest.fail("faceapi did not start")
    yield
    server.should_exit = True
    t.join(timeout=10)


@pytest.fixture(scope="module")
def client(face_server):
    with TestClient(att_app) as c:
        yield c


@pytest.fixture(scope="module")
def faces():
    return fetch_olivetti_faces(shuffle=False)


def photo(faces, pid, idx) -> str:
    b, a = augment_params(pid * 100 + idx)
    jpg = to_jpeg_bytes(face_to_frame(faces.images[faces.target == pid][idx],
                                      brightness=b, rotate_deg=a))
    return "data:image/jpeg;base64," + base64.b64encode(jpg).decode()


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── the flow ─────────────────────────────────────────────────────────────────
STATE: dict = {}


def test_face_service_reachable(client):
    h = client.get("/api/health").json()
    assert h["face_service"]["status"] == "ok", h


def test_admin_login(client):
    r = client.post("/api/auth/login",
                    json={"email": "admin@example.com", "password": "admin123"})
    assert r.status_code == 200, r.text
    STATE["admin"] = r.json()["token"]
    assert r.json()["user"]["role"] == "admin"


def test_bad_login_rejected(client):
    assert client.post("/api/auth/login",
                       json={"email": "admin@example.com", "password": "wrong"}).status_code == 401


def test_create_users_and_course(client):
    a = auth(STATE["admin"])
    r = client.post("/api/users", headers=a, json={
        "email": "teacher@example.com", "password": "teach123",
        "full_name": "Terry Teacher", "role": "teacher"})
    assert r.status_code == 201, r.text

    STATE["students"] = []
    for i, pid in enumerate([3, 11, 27]):
        r = client.post("/api/users", headers=a, json={
            "email": f"s{i}@example.com", "password": "pass123",
            "full_name": f"Student {i}", "role": "student",
            "student_number": f"STU-{i:03d}"})
        assert r.status_code == 201, r.text
        STATE["students"].append({"id": r.json()["id"], "pid": pid, "email": f"s{i}@example.com"})

    r = client.post("/api/courses", headers=a, json={"code": "CS101", "name": "Computing"})
    assert r.status_code == 201, r.text
    STATE["course"] = r.json()["id"]

    for s in STATE["students"]:
        assert client.post(f"/api/courses/{STATE['course']}/enrol?student_id={s['id']}",
                           headers=a).status_code == 200


def test_students_enrol_faces(client, faces):
    """Each student logs in and enrols their own face through the API."""
    for s in STATE["students"]:
        tok = client.post("/api/auth/login",
                          json={"email": s["email"], "password": "pass123"}).json()["token"]
        s["token"] = tok
        r = client.post("/api/face/enroll", headers=auth(tok),
                        json={"images": [photo(faces, s["pid"], i) for i in range(3)]})
        assert r.status_code == 200, r.text
        assert r.json()["enrolled"] is True
        assert r.json()["embeddings_added"] >= 3
    assert client.get("/api/auth/me", headers=auth(STATE["students"][0]["token"])
                      ).json()["face_enrolled"] is True


def test_checkin_requires_open_session(client, faces):
    s = STATE["students"][0]
    r = client.post("/api/checkin", headers=auth(s["token"]),
                    json={"image": photo(faces, s["pid"], 7)})
    assert r.json()["status"] == "no_open_session", r.text


def test_open_session_and_face_checkin(client, faces):
    teacher = client.post("/api/auth/login",
                          json={"email": "teacher@example.com", "password": "teach123"}
                          ).json()["token"]
    STATE["teacher"] = teacher
    r = client.post("/api/sessions", headers=auth(teacher),
                    json={"course_id": STATE["course"], "title": "Week 1"})
    assert r.status_code == 201, r.text
    STATE["session"] = r.json()["id"]

    # student 0 checks in with a HELD-OUT photo (not one used for enrolment)
    s = STATE["students"][0]
    r = client.post("/api/checkin", headers=auth(s["token"]),
                    json={"image": photo(faces, s["pid"], 7)}).json()
    assert r["status"] == "marked", r
    assert r["student"]["id"] == s["id"]
    assert r["confidence"] > 0.4


def test_duplicate_checkin_blocked(client, faces):
    s = STATE["students"][0]
    r = client.post("/api/checkin", headers=auth(s["token"]),
                    json={"image": photo(faces, s["pid"], 8)}).json()
    assert r["status"] == "already_marked", r


def test_kiosk_identifies_other_student(client, faces):
    s = STATE["students"][1]
    r = client.post("/api/kiosk/checkin", headers=auth(STATE["teacher"]),
                    json={"image": photo(faces, s["pid"], 7),
                          "session_id": STATE["session"]}).json()
    assert r["status"] == "marked" and r["student"]["id"] == s["id"], r


def test_unknown_face_rejected(client, faces):
    """A person who never enrolled must not be matched to anyone."""
    r = client.post("/api/kiosk/checkin", headers=auth(STATE["teacher"]),
                    json={"image": photo(faces, 39, 0),
                          "session_id": STATE["session"]}).json()
    assert r["status"] in ("unknown", "low_confidence"), r


def test_student_cannot_use_staff_endpoints(client, faces):
    s = STATE["students"][0]
    assert client.post("/api/kiosk/checkin", headers=auth(s["token"]),
                       json={"image": photo(faces, s["pid"], 9)}).status_code == 403
    assert client.get("/api/stats", headers=auth(s["token"])).status_code == 403


def test_manual_mark_and_session_view(client):
    third = STATE["students"][2]
    r = client.post("/api/attendance/manual", headers=auth(STATE["teacher"]),
                    json={"session_id": STATE["session"], "student_id": third["id"]})
    assert r.status_code == 200, r.text
    d = client.get(f"/api/sessions/{STATE['session']}/attendance",
                   headers=auth(STATE["teacher"])).json()
    assert len(d["present"]) == 3 and len(d["absent"]) == 0
    assert {p["method"] for p in d["present"]} == {"face", "manual"}


def test_student_portal_percentages(client):
    s = STATE["students"][0]
    d = client.get("/api/me/attendance", headers=auth(s["token"])).json()
    assert d["courses"][0]["code"] == "CS101"
    assert d["courses"][0]["percentage"] == 100.0
    assert len(d["history"]) == 1


def test_reports_and_csv(client):
    a = auth(STATE["teacher"])
    rep = client.get(f"/api/reports/course/{STATE['course']}", headers=a).json()
    assert len(rep["students"]) == 3
    assert all(s["percentage"] == 100.0 for s in rep["students"])
    csv_r = client.get(f"/api/reports/course/{STATE['course']}/export.csv", headers=a)
    assert csv_r.status_code == 200
    assert "sessions_attended" in csv_r.text
    assert len(csv_r.text.strip().splitlines()) == 4      # header + 3 students


def test_close_session_blocks_checkin(client, faces):
    client.post(f"/api/sessions/{STATE['session']}/close", headers=auth(STATE["teacher"]))
    s = STATE["students"][0]
    r = client.post("/api/checkin", headers=auth(s["token"]),
                    json={"image": photo(faces, s["pid"], 9)}).json()
    assert r["status"] in ("no_open_session", "session_closed"), r
