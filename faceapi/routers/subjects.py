"""Gallery management routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from faceapi.deps import get_store, require_key
from faceapi.schemas import SubjectInfo
from faceapi.store import SubjectStore

router = APIRouter(tags=["subjects"], dependencies=[Depends(require_key)])


@router.get("/subjects", response_model=list[SubjectInfo])
async def list_subjects(store: SubjectStore = Depends(get_store)):
    return [SubjectInfo(**s) for s in store.list_subjects()]


@router.delete("/subjects/{subject_id}")
async def delete_subject(subject_id: str, store: SubjectStore = Depends(get_store)):
    if not store.delete_subject(subject_id):
        raise HTTPException(404, "Subject not found")
    return {"deleted": subject_id}
