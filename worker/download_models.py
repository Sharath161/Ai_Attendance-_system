"""Download the models used by ``worker.model_adapter``.

Usage:
    python -m worker.download_models

Files are written to ``work/models/`` (paths from ``core.config.Settings``).
Re-running skips files that already exist.

Models downloaded
-----------------
* face_detection_yunet_2023mar.onnx  — YuNet face detector, 5-point landmarks
                                       (OpenCV Zoo, ~232 KB)
* w600k_mbf.onnx                     — ArcFace MobileNet recognition model, 512-d
                                       (InsightFace buffalo_sc pack, ~16 MB)
"""
from __future__ import annotations

import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

import requests

from core.config import get_settings

_OPENCV_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"

_DETECTION_URL = (
    f"{_OPENCV_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

# InsightFace buffalo_sc pack — contains w600k_mbf.onnx (ArcFace MobileNet, 512-d)
_INSIGHTFACE_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
)
_ARCFACE_INNER = "w600k_mbf.onnx"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FaceAttendance/1.0; +download_models)"
}


def _download_file(url: str, destination: Path) -> None:
    if destination.exists():
        print(f"skip (exists): {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    try:
        r = requests.get(url, headers=_HEADERS, stream=True, timeout=120)
        r.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in r.iter_content(chunk_size=256 * 1024):
                f.write(chunk)
        print(f"  -> {destination}  ({destination.stat().st_size:,} bytes)")
    except Exception:
        # Fall back to urllib for environments where requests is unavailable
        destination.unlink(missing_ok=True)
        print(f"  retrying with urllib...")
        urllib.request.urlretrieve(url, destination)  # noqa: S310
        print(f"  -> {destination}  ({destination.stat().st_size:,} bytes)")


def _download_from_zip(url: str, inner_name: str, destination: Path) -> None:
    if destination.exists():
        print(f"skip (exists): {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}  (extracting {inner_name})")
    r = requests.get(url, headers=_HEADERS, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(BytesIO(r.content)) as zf:
        matches = [n for n in zf.namelist() if n.endswith(inner_name)]
        if not matches:
            raise RuntimeError(
                f"{inner_name} not found in zip from {url}. "
                f"Available: {zf.namelist()}"
            )
        with zf.open(matches[0]) as src, open(destination, "wb") as dst:
            dst.write(src.read())
    print(f"  -> {destination}  ({destination.stat().st_size:,} bytes)")


def main() -> None:
    settings = get_settings()

    _download_file(_DETECTION_URL, Path(settings.face_detection_model_path))
    _download_from_zip(
        _INSIGHTFACE_URL, _ARCFACE_INNER, Path(settings.face_recognition_model_path)
    )

    print("\nAll models ready.")
    print(f"  Detection  : {settings.face_detection_model_path}")
    print(f"  Recognition: {settings.face_recognition_model_path}")


if __name__ == "__main__":
    main()
