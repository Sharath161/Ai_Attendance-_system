from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute project root — works regardless of cwd
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_WORK_DIR = _PROJECT_ROOT / "work"


class Settings(BaseSettings):
    database_url: str = f"sqlite+aiosqlite:///{_WORK_DIR / 'attendance.db'}"
    spool_dir: Path = _WORK_DIR / "spool"
    max_upload_bytes: int = 1_048_576

    api_db_pool_size: int = 5
    api_db_max_overflow: int = 0
    db_pool_timeout_seconds: int = 3
    db_pool_recycle_seconds: int = 1800

    worker_poll_interval_seconds: float = 1.0

    # ── model pipeline ────────────────────────────────────────────────────────
    # Detection: YuNet (OpenCV Zoo) — fast CPU inference, 5-point landmarks
    face_detection_model_path: Path = _WORK_DIR / "models" / "face_detection_yunet_2023mar.onnx"
    detection_score_threshold: float = 0.60

    # Recognition: ArcFace MobileNet (InsightFace buffalo_sc, 512-d)
    face_recognition_model_path: Path = _WORK_DIR / "models" / "w600k_mbf.onnx"
    worker_model_version: str = "arcface-mbf"
    embedding_dimensions: int = 512

    # ── matching ──────────────────────────────────────────────────────────────
    # ArcFace cosine-similarity threshold (L2-normalised dot product).
    # Range: 0.0–1.0. Typical values: 0.28–0.45.
    # Raise to reduce false positives; lower to reduce false negatives.
    match_threshold: float = 0.35

    # Optional per-race thresholds; empty dict → use global match_threshold.
    # Keys use FairFace race labels (lowercase, underscored):
    #   white, black, indian, east_asian, southeast_asian, middle_eastern, latino_hispanic
    race_thresholds: dict[str, float] = {}

    auto_create_tables: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    def threshold_for(self, race: str | None) -> float:
        """Return the effective match threshold for a demographic group."""
        if race and self.race_thresholds:
            return self.race_thresholds.get(race.lower().replace(" ", "_"), self.match_threshold)
        return self.match_threshold


@lru_cache
def get_settings() -> Settings:
    return Settings()
