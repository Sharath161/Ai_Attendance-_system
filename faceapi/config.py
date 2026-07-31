"""Configuration for the face-recognition service (env prefix FACEAPI_)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parent.parent
_WORK = _ROOT / "work"
_MODELS = _WORK / "models"


class Settings(BaseSettings):
    # ── models ────────────────────────────────────────────────────────────────
    detection_model_path: Path = _MODELS / "face_detection_yunet_2023mar.onnx"
    recognition_model_full: Path = _MODELS / "w600k_mbf.onnx"
    recognition_model_int8: Path = _MODELS / "w600k_mbf.int8.onnx"

    # Low-resource: the full 'mbf' model is already tiny (13.6 MB, ~5 ms/face) and
    # is the recommended edge/CPU choice. INT8 is OPT-IN: ~3.9x smaller (memory win
    # for flash/RAM-limited boards) but slower on x86 CPUs — benchmark on your target
    # (ARM SBCs often speed up). Set intra_op_threads to your core count on a Pi.
    use_int8: bool = False               # opt-in quantized model (memory-constrained edge)
    intra_op_threads: int = 0            # 0 = ORT auto; set e.g. 4 on a Pi 4

    detection_score_threshold: float = 0.60
    enable_tta: bool = False             # flip TTA — +accuracy, 2x embed cost

    # ── matching ──────────────────────────────────────────────────────────────
    # Calibrated cosine threshold (set from faceapi.calibrate on your domain).
    match_threshold: float = 0.42
    max_faces: int = 10
    max_upload_bytes: int = 4_194_304    # 4 MB

    # ── store & auth ──────────────────────────────────────────────────────────
    db_path: Path = _WORK / "faceapi.db"
    # Comma-separated API keys accepted in the X-API-Key header.
    # Empty => auth disabled (dev only).
    api_keys: str = ""
    # Comma-separated allowed CORS origins; empty => "*" (lock down in production).
    cors_origins: str = ""

    model_config = SettingsConfigDict(env_prefix="FACEAPI_", env_file=".env",
                                      env_file_encoding="utf-8", extra="ignore")

    def recognition_model(self) -> Path:
        if self.use_int8 and self.recognition_model_int8.exists():
            return self.recognition_model_int8
        return self.recognition_model_full

    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
