"""Pytest smoke coverage for the Face API (in-process TestClient)."""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import pytest

_tmp = Path(tempfile.gettempdir()) / "faceapi_pytest.db"
_tmp.unlink(missing_ok=True)
os.environ["FACEAPI_DB_PATH"] = str(_tmp)
os.environ["FACEAPI_API_KEYS"] = ""

from fastapi.testclient import TestClient
from sklearn.datasets import fetch_olivetti_faces

from faceapi.api import app
from tests.stress.image_seed import face_to_frame, to_jpeg_bytes


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c
    try:
        app.state.store.close(); _tmp.unlink(missing_ok=True)
    except Exception:
        pass


@pytest.fixture(scope="module")
def data():
    return fetch_olivetti_faces(shuffle=False)


def _jpg(face):
    return to_jpeg_bytes(face_to_frame(face))


def test_health(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model"].endswith(".onnx")


def test_enroll_and_identify(client, data):
    for pid in (5, 15):
        files = [("images", (f"e{i}.jpg", io.BytesIO(_jpg(data.images[data.target == pid][i])), "image/jpeg"))
                 for i in range(5)]
        r = client.post("/enroll", data={"subject_id": f"S{pid}", "name": f"P{pid}"}, files=files)
        assert r.status_code == 200 and r.json()["embeddings_added"] == 5

    # held-out image of person 5 should identify as S5
    q = {"image": ("q.jpg", io.BytesIO(_jpg(data.images[data.target == 5][8])), "image/jpeg")}
    r = client.post("/identify", files=q).json()
    assert r["status"] == "matched" and r["match"]["subject_id"] == "S5"


def test_embed_dim(client, data):
    q = {"image": ("q.jpg", io.BytesIO(_jpg(data.images[data.target == 5][0])), "image/jpeg")}
    e = client.post("/embed", files=q).json()
    assert e["face_count"] == 1 and e["dim"] == 512 and len(e["embedding"]) == 512


def test_auth_enforced(data):
    os.environ["FACEAPI_API_KEYS"] = "secret123"
    from faceapi.config import get_settings
    get_settings.cache_clear()
    with TestClient(app) as c:
        q = {"image": ("q.jpg", io.BytesIO(_jpg(data.images[data.target == 5][0])), "image/jpeg")}
        assert c.post("/embed", files=q).status_code == 401
        assert c.post("/embed", files=q, headers={"X-API-Key": "secret123"}).status_code == 200
    os.environ["FACEAPI_API_KEYS"] = ""
    get_settings.cache_clear()
