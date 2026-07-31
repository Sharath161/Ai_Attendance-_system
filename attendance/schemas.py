"""Request/response models for the attendance API."""
from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


# ── auth ──────────────────────────────────────────────────────────────────────
class LoginBody(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    token: str
    user: "UserOut"


class UserOut(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    student_number: str | None = None
    face_enrolled: bool = False


class RegisterBody(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    full_name: str
    role: str = "student"
    student_number: str | None = None


# ── courses & sessions ────────────────────────────────────────────────────────
class CourseBody(BaseModel):
    code: str
    name: str
    teacher_id: str | None = None


class SessionBody(BaseModel):
    course_id: str
    title: str | None = None


# ── face & check-in ───────────────────────────────────────────────────────────
class ImageBody(BaseModel):
    image: str                       # base64 or data: URI


class EnrolFaceBody(BaseModel):
    images: list[str]


class CheckinBody(ImageBody):
    session_id: str | None = None


class ManualMarkBody(BaseModel):
    session_id: str
    student_id: str


# ── reports ───────────────────────────────────────────────────────────────────
class CoursePercentage(BaseModel):
    course_id: str
    code: str
    name: str
    total: int
    attended: int
    percentage: float


class StudentReportRow(BaseModel):
    student_id: str
    full_name: str
    student_number: str | None = None
    total: int
    attended: int
    percentage: float


TokenResponse.model_rebuild()
