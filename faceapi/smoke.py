"""In-process smoke test of the Face API (no server needed).

    python -m faceapi.smoke

Uses a throwaway temp DB, enrols 3 distinct people from Olivetti, then exercises
identify / verify / embed / subjects / health.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

# isolate DB + disable auth BEFORE importing the app/settings
_tmp = Path(tempfile.gettempdir()) / "faceapi_smoke.db"
_tmp.unlink(missing_ok=True)
os.environ["FACEAPI_DB_PATH"] = str(_tmp)
os.environ["FACEAPI_API_KEYS"] = ""
os.environ["FACEAPI_MATCH_THRESHOLD"] = "0.42"

from fastapi.testclient import TestClient
from sklearn.datasets import fetch_olivetti_faces

from faceapi.api import app
from faceapi._testutil import face_to_frame, to_jpeg_bytes, augment_params


def jpg(face, pid, idx):
    b, a = augment_params(pid * 100 + idx)
    return to_jpeg_bytes(face_to_frame(face, brightness=b, rotate_deg=a))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data = fetch_olivetti_faces(shuffle=False)
    people = [3, 11, 27]   # three distinct identities
    print("=" * 58)
    print("  FACE API — in-process smoke test")
    print("=" * 58)

    with TestClient(app) as c:
        h = c.get("/health").json()
        print(f"  health: model={h['model']} int8={h['int8']} tta={h['tta']} thr={h['threshold']}")

        # enrol
        for pid in people:
            files = [("images", (f"e{i}.jpg", io.BytesIO(jpg(data.images[data.target == pid][i], pid, i)),
                                  "image/jpeg")) for i in range(5)]
            r = c.post("/enroll", data={"subject_id": f"SUBJ-{pid}", "name": f"Person {pid}"}, files=files)
            b = r.json()
            print(f"  enroll SUBJ-{pid}: +{b['embeddings_added']} emb (rejected {b['faces_rejected']})")

        print(f"  subjects: {[s['subject_id'] for s in c.get('/subjects').json()]}")

        # identify held-out images (idx 7,8 held out)
        print("\n  identify (held-out):")
        correct = total = 0
        for pid in people:
            for i in (7, 8):
                files = {"image": (f"q.jpg", io.BytesIO(jpg(data.images[data.target == pid][i], pid, i)), "image/jpeg")}
                r = c.post("/identify", files=files).json()
                m = r.get("match")
                ok = m and m["subject_id"] == f"SUBJ-{pid}"
                correct += bool(ok); total += 1
                who = f"{m['subject_id']} ({m['score']:.3f})" if m else r["status"]
                print(f"    true SUBJ-{pid} -> {r['status']:<9} {who}  {'OK' if ok else 'X'}")
        print(f"  identify accuracy: {correct}/{total}")

        # verify 1:1 (genuine + impostor)
        gv = c.post("/verify", data={"subject_id": "SUBJ-3"},
                    files={"image": ("q.jpg", io.BytesIO(jpg(data.images[data.target == 3][9], 3, 9)), "image/jpeg")}).json()
        iv = c.post("/verify", data={"subject_id": "SUBJ-3"},
                    files={"image": ("q.jpg", io.BytesIO(jpg(data.images[data.target == 11][9], 11, 9)), "image/jpeg")}).json()
        print(f"\n  verify SUBJ-3 vs own photo   : {gv['status']}  score={gv.get('score')}")
        print(f"  verify SUBJ-3 vs other person: {iv['status']}  score={iv.get('score')}")

        # embed
        e = c.post("/embed", files={"image": ("q.jpg", io.BytesIO(jpg(data.images[data.target == 3][0], 3, 0)), "image/jpeg")}).json()
        print(f"\n  embed: face_count={e['face_count']} dim={e['dim']} model={e['model']}")

        # delete + auth check
        print(f"  delete SUBJ-27: {c.delete('/subjects/SUBJ-27').json()}")

    try:
        _tmp.unlink(missing_ok=True)
    except Exception:
        pass
    print("\n  smoke test complete.")


if __name__ == "__main__":
    main()
