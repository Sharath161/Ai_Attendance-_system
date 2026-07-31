"""Course and enrolment routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from attendance.auth import current_user, require_staff
from attendance.db import Database
from attendance.deps import get_db, user_out
from attendance.schemas import CourseBody

router = APIRouter(prefix="/courses", tags=["courses"])


@router.post("", status_code=201)
async def create_course(body: CourseBody, db: Database = Depends(get_db),
                        _=Depends(require_staff)):
    if db.one("SELECT 1 FROM courses WHERE code=?", (body.code.strip(),)):
        raise HTTPException(409, "Course code already exists")
    cid = db.create_course(body.code, body.name, body.teacher_id)
    return {"id": cid, "code": body.code, "name": body.name}


@router.get("")
async def list_courses(db: Database = Depends(get_db), user=Depends(current_user)):
    rows = db.courses_for_student(user["id"]) if user["role"] == "student" else db.list_courses()
    return [dict(r) for r in rows]


@router.post("/{course_id}/enrol")
async def enrol_student(course_id: str, student_id: str, db: Database = Depends(get_db),
                        _=Depends(require_staff)):
    if db.user_by_id(student_id) is None:
        raise HTTPException(404, "Student not found")
    db.enrol(course_id, student_id)
    return {"enrolled": student_id, "course": course_id}


@router.delete("/{course_id}/enrol/{student_id}")
async def unenrol_student(course_id: str, student_id: str, db: Database = Depends(get_db),
                          _=Depends(require_staff)):
    db.unenrol(course_id, student_id)
    return {"unenrolled": student_id}


@router.get("/{course_id}/students")
async def course_students(course_id: str, db: Database = Depends(get_db),
                          _=Depends(require_staff)):
    return [user_out(u) for u in db.course_students(course_id)]
