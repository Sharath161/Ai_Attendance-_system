"""Fetch the ONNX models the service needs.

    python -m faceapi.download_models

Writes into the paths configured in faceapi.config (default work/models/):
  * face_detection_yunet_2023mar.onnx  — YuNet detector, 5-point landmarks (~232 KB)
  * w600k_mbf.onnx                     — ArcFace MobileNet, 512-d (~13.6 MB)

Re-running skips files that already exist.
"""
from __future__ import annotations

import sys
import zipfile
from io import BytesIO
from pathlib import Path

import requests

from faceapi.config import get_settings

_DETECTION_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
                  "face_detection_yunet/face_detection_yunet_2023mar.onnx")
# InsightFace buffalo_sc pack contains w600k_mbf.onnx
_RECOGNITION_ZIP = ("https://github.com/deepinsight/insightface/releases/download/"
                    "v0.7/buffalo_sc.zip")
_RECOGNITION_INNER = "w600k_mbf.onnx"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; faceapi/1.0)"}


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"skip (exists): {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    r = requests.get(url, headers=_HEADERS, stream=True, timeout=180)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=256 * 1024):
            f.write(chunk)
    print(f"  -> {dest} ({dest.stat().st_size:,} bytes)")


def _download_from_zip(url: str, inner: str, dest: Path) -> None:
    if dest.exists():
        print(f"skip (exists): {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} (extracting {inner})")
    r = requests.get(url, headers=_HEADERS, timeout=300)
    r.raise_for_status()
    with zipfile.ZipFile(BytesIO(r.content)) as zf:
        match = [n for n in zf.namelist() if n.endswith(inner)]
        if not match:
            raise RuntimeError(f"{inner} not found in {url}")
        with zf.open(match[0]) as src, open(dest, "wb") as out:
            out.write(src.read())
    print(f"  -> {dest} ({dest.stat().st_size:,} bytes)")


def main() -> None:
    s = get_settings()
    _download(_DETECTION_URL, Path(s.detection_model_path))
    _download_from_zip(_RECOGNITION_ZIP, _RECOGNITION_INNER, Path(s.recognition_model_full))
    print("\nModels ready.")
    print(f"  detection  : {s.detection_model_path}")
    print(f"  recognition: {s.recognition_model_full}")


if __name__ == "__main__":
    sys.exit(main())
