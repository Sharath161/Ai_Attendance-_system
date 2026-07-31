"""Authentication and user management routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from attendance import auth
from attendance.auth import current_user, hash_password, require_admin, require_staff, verify_password
from attendance.config import Settings, get_settings
from attendance.db import Database
from attendance.deps import get_db, get_face, user_out
from attendance.faceclient import FaceClient
from attendance.schemas import LoginBody, RegisterBody

router = APIRouter(tags=["auth"])


@router.post("/auth/login")
async def login(body: LoginBody, db: Database = Depends(get_db),
                settings: Settings = Depends(get_settings)):
    user = db.user_by_email(body.email)
    if user is None or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    return {"token": auth.create_token(user["id"], user["role"], settings),
            "user": user_out(user)}


@router.get("/auth/me")
async def me(user=Depends(current_user)):
    return user_out(user)


@router.post("/users", status_code=201)
async def create_user(body: RegisterBody, db: Database = Depends(get_db),
                      _=Depends(require_admin)):
    if body.role not in ("student", "teacher", "admin"):
        raise HTTPException(400, "role must be student, teacher or admin")
    if db.user_by_email(body.email):
        raise HTTPException(409, "Email already registered")
    uid = db.create_user(body.email, hash_password(body.password),
                         body.full_name, body.role, body.student_number)
    return user_out(db.user_by_id(uid))


@router.get("/users")
async def list_users(role: str | None = None, db: Database = Depends(get_db),
                     _=Depends(require_staff)):
    return [user_out(u) for u in db.list_users(role)]


@router.delete("/users/{user_id}")
async def delete_user(user_id: str, db: Database = Depends(get_db),
                      face: FaceClient = Depends(get_face), _=Depends(require_admin)):
    user = db.user_by_id(user_id)
    if user is None:
        raise HTTPException(404, "User not found")
    await face.delete_subject(user["face_subject_id"])
    db.run("DELETE FROM users WHERE id=?", (user_id,))
    return {"deleted": user_id}
