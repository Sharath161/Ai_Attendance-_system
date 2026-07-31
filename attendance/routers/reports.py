"""Student portal and staff reporting routes."""
from __future__ import annotations

import csv
import io

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from attendance.auth import require_staff, require_student
from attendance.db import Database
from attendance.deps import get_db

router = APIRouter(tags=["reports"])


def _pct(attended: int, total: int) -> float:
    return round(attended / total * 100, 1) if total else 0.0


@router.get("/me/attendance")
async def my_attendance(db: Database = Depends(get_db), user=Depends(require_student)):
    courses = [{"course_id": r["id"], "code": r["code"], "name": r["name"],
                "total": r["total"], "attended": r["attended"],
                "percentage": _pct(r["attended"], r["total"])}
               for r in db.student_percentages(user["id"])]
    return {"courses": courses,
            "history": [dict(r) for r in db.student_history(user["id"])]}


@router.get("/reports/course/{course_id}")
async def course_report(course_id: str, db: Database = Depends(get_db),
                        _=Depends(require_staff)):
    students = [{"student_id": r["id"], "full_name": r["full_name"],
                 "student_number": r["student_number"], "total": r["total"],
                 "attended": r["attended"], "percentage": _pct(r["attended"], r["total"])}
                for r in db.course_report(course_id)]
    return {"course_id": course_id, "students": students}


@router.get("/reports/course/{course_id}/export.csv")
async def export_csv(course_id: str, db: Database = Depends(get_db),
                     _=Depends(require_staff)):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["student_number", "full_name", "sessions_attended",
                     "sessions_total", "percentage"])
    for r in db.course_report(course_id):
        writer.writerow([r["student_number"] or "", r["full_name"], r["attended"],
                         r["total"], _pct(r["attended"], r["total"])])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="attendance_{course_id}.csv"'})


@router.get("/stats")
async def stats(db: Database = Depends(get_db), _=Depends(require_staff)):
    return db.stats()
