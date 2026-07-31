"""Attendance service configuration (env prefix ATT_)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parent.parent
_WORK = _ROOT / "work"


class Settings(BaseSettings):
    # ── storage ───────────────────────────────────────────────────────────────
    db_path: Path = _WORK / "attendance.db"

    # ── face recognition service (separate microservice) ─────────────────────
    faceapi_url: str = "http://localhost:8080"
    faceapi_key: str = ""            # sent as X-API-Key if set
    faceapi_timeout: float = 30.0

    # Minimum confidence to auto-accept a face check-in. The faceapi service
    # applies its own calibrated threshold; this is an extra business-level gate.
    min_checkin_confidence: float = 0.45

    # ── auth ──────────────────────────────────────────────────────────────────
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 720          # 12 h

    # Bootstrap admin created on first start if no users exist.
    bootstrap_admin_email: str = "admin@example.com"
    bootstrap_admin_password: str = "admin123"

    # ── behaviour ─────────────────────────────────────────────────────────────
    checkin_cooldown_minutes: int = 30     # block duplicate check-ins per session
    max_upload_bytes: int = 4_194_304
    cors_origins: str = ""                 # empty => "*"

    model_config = SettingsConfigDict(env_prefix="ATT_", env_file=".env",
                                      env_file_encoding="utf-8", extra="ignore")

    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()] or ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
