"""Pydantic response models for the face API."""
from __future__ import annotations

from pydantic import BaseModel


class BBox(BaseModel):
    x: int; y: int; w: int; h: int


class Candidate(BaseModel):
    subject_id: str
    name: str | None = None
    score: float


class EnrollResponse(BaseModel):
    subject_id: str
    name: str | None = None
    embeddings_added: int
    total_embeddings: int
    faces_rejected: int


class IdentifyResponse(BaseModel):
    status: str                       # matched | no_match | no_face | multiple_faces
    match: Candidate | None = None
    candidates: list[Candidate] = []
    face_box: BBox | None = None
    threshold: float


class VerifyResponse(BaseModel):
    status: str                       # verified | rejected | no_face | multiple_faces | unknown_subject
    is_match: bool | None = None
    score: float | None = None
    threshold: float


class EmbedResponse(BaseModel):
    face_count: int
    embedding: list[float] | None = None
    face_box: BBox | None = None
    model: str
    dim: int


class SubjectInfo(BaseModel):
    subject_id: str
    name: str | None = None
    embeddings: int
    metadata: dict = {}


class HealthResponse(BaseModel):
    status: str
    model: str
    int8: bool
    tta: bool
    threshold: float
    subjects: int
    embeddings: int
