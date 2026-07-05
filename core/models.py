import enum
import secrets
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ── ESP32 device registry ─────────────────────────────────────────────────────

class DeviceStatus(str, enum.Enum):
    active  = "active"
    offline = "offline"
    fault   = "fault"


class ESP32Device(Base):
    """One row per physical ESP32-CAM unit deployed in a room."""
    __tablename__ = "esp32_devices"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Human-readable name, e.g. "LT-204-Door"
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    # MAC address of the ESP32 WiFi interface, stored as XX:XX:XX:XX:XX:XX
    device_mac: Mapped[str | None] = mapped_column(String(17), unique=True, nullable=True)
    # Room / location label, e.g. "Building A - Room 204"
    room_label: Mapped[str | None] = mapped_column(String(150), nullable=True)
    # Secret key the ESP32 sends in X-Device-Key header.
    # Generated once on registration; never changes.
    api_key: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False,
        default=lambda: secrets.token_hex(32),
    )
    firmware_version: Mapped[str | None] = mapped_column(String(20), nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[DeviceStatus] = mapped_column(
        Enum(DeviceStatus, name="device_status"),
        default=DeviceStatus.active,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class StudentStatus(str, enum.Enum):
    active = "active"
    pending_embedding = "pending_embedding"
    embedding_failed = "embedding_failed"


class JobStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    complete = "complete"
    failed = "failed"


class AttendanceStatus(str, enum.Enum):
    recognized = "recognized"
    unknown = "unknown"
    low_confidence = "low_confidence"
    no_face_detected = "no_face_detected"
    multiple_faces_detected = "multiple_faces_detected"


class Student(Base):
    __tablename__ = "students"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    class_id: Mapped[str | None] = mapped_column(String(64), index=True)
    status: Mapped[StudentStatus] = mapped_column(
        Enum(StudentStatus, name="student_status"),
        default=StudentStatus.pending_embedding,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    embeddings: Mapped[list["StudentEmbedding"]] = relationship(back_populates="student")


class StudentEmbedding(Base):
    __tablename__ = "student_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("students.id"), nullable=False, index=True)
    embedding: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    image_quality_score: Mapped[float | None] = mapped_column(Float)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    # Optional demographic label for fairness tracking.
    # Uses FairFace race labels: white, black, indian, east_asian,
    # southeast_asian, middle_eastern, latino_hispanic
    demographic_group: Mapped[str | None] = mapped_column(String(50), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    student: Mapped[Student] = relationship(back_populates="embeddings")


class ImageJob(Base):
    __tablename__ = "image_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_path: Mapped[str] = mapped_column(Text, nullable=False)
    # Legacy string camera_id kept for backwards compat; new rows also set device_id FK.
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # FK to the registered ESP32 device that captured this image (nullable for old rows).
    device_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("esp32_devices.id"), nullable=True, index=True
    )
    class_id: Mapped[str | None] = mapped_column(String(64), index=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status"),
        default=JobStatus.pending,
        nullable=False,
        index=True,
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AttendanceEvent(Base):
    __tablename__ = "attendance_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("students.id"), index=True)
    # Closest match even when below threshold, for the human review queue.
    candidate_student_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("students.id"), index=True)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("image_jobs.id"), nullable=False, index=True)
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False)
    class_id: Mapped[str | None] = mapped_column(String(64), index=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    status: Mapped[AttendanceStatus] = mapped_column(Enum(AttendanceStatus, name="attendance_status"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class FairnessMetric(Base):
    """Stores results of per-race fairness validation runs."""

    __tablename__ = "fairness_metrics"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    test_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    dataset: Mapped[str] = mapped_column(String(64), nullable=False)        # e.g. "fairface", "olivetti"
    race_group: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    detection_rate: Mapped[float | None] = mapped_column(Float)             # fraction of images where face found
    precision: Mapped[float | None] = mapped_column(Float)
    recall: Mapped[float | None] = mapped_column(Float)
    f1: Mapped[float | None] = mapped_column(Float)
    threshold_used: Mapped[float] = mapped_column(Float, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
