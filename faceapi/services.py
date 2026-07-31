"""Recognition operations shared by the multipart and JSON routes."""
from __future__ import annotations

from fastapi import HTTPException

from faceapi import matching
from faceapi.config import Settings
from faceapi.engine import FaceEngine
from faceapi.fewshot import amplified_embeddings
from faceapi.schemas import (BBox, Candidate, EmbedResponse, EnrollResponse,
                             IdentifyResponse, VerifyResponse)
from faceapi.store import SubjectStore


def embed_face(engine: FaceEngine, frame, settings: Settings) -> EmbedResponse:
    dets = engine.detect(frame, embed=True, max_faces=settings.max_faces)
    if not dets:
        return EmbedResponse(face_count=0, embedding=None,
                             model=engine.recognition_model_name, dim=512)
    d = dets[0]
    return EmbedResponse(face_count=len(dets),
                         embedding=[round(float(x), 6) for x in d.embedding],
                         face_box=BBox(**d.bbox),
                         model=engine.recognition_model_name, dim=512)


def identify_face(engine: FaceEngine, store: SubjectStore, frame,
                  settings: Settings) -> IdentifyResponse:
    dets = engine.detect(frame, embed=True, max_faces=settings.max_faces)
    thr = settings.match_threshold
    if not dets:
        return IdentifyResponse(status="no_face", threshold=thr)
    if len(dets) > 1:
        return IdentifyResponse(status="multiple_faces",
                                face_box=BBox(**dets[0].bbox), threshold=thr)
    res = matching.identify(dets[0].embedding, store.gallery(), thr)
    match = Candidate(**res["match"]) if res["match"] else None
    return IdentifyResponse(status="matched" if match else "no_match", match=match,
                            candidates=[Candidate(**c) for c in res["candidates"]],
                            face_box=BBox(**dets[0].bbox), threshold=thr)


def verify_face(engine: FaceEngine, store: SubjectStore, subject_id: str, frame,
                settings: Settings) -> VerifyResponse:
    thr = settings.match_threshold
    mat = store.subject_vectors(subject_id)
    if mat is None:
        return VerifyResponse(status="unknown_subject", threshold=thr)
    dets = engine.detect(frame, embed=True, max_faces=settings.max_faces)
    if not dets:
        return VerifyResponse(status="no_face", threshold=thr)
    if len(dets) > 1:
        return VerifyResponse(status="multiple_faces", threshold=thr)
    r = matching.verify(dets[0].embedding, mat, thr)
    return VerifyResponse(status="verified" if r["is_match"] else "rejected",
                          is_match=r["is_match"], score=round(r["score"], 4), threshold=thr)


def enroll_subject(engine: FaceEngine, store: SubjectStore, subject_id: str,
                   name: str | None, metadata: dict | None, frames: list,
                   settings: Settings) -> EnrollResponse:
    vectors, rejected = [], 0
    for frame in frames:
        crops = engine.align_crops(frame, max_faces=settings.max_faces)
        if len(crops) != 1:
            rejected += 1
            continue
        if settings.enroll_amplify:
            # few-shot amplifier: one photo -> ~6 augmented prototype embeddings
            vectors.extend(amplified_embeddings(engine, crops[0]["crop"], settings.embed_batch))
        else:
            vectors.append(engine.embed_crops([crops[0]["crop"]])[0])

    if not vectors:
        raise HTTPException(422, "No usable single-face images "
                                 "(each photo needs exactly one face)")

    store.upsert_subject(subject_id, name, metadata)
    added = store.add_embeddings(subject_id, vectors, engine.recognition_model_name)
    existing = store.subject_vectors(subject_id)
    return EnrollResponse(subject_id=subject_id, name=name, embeddings_added=added,
                          total_embeddings=0 if existing is None else len(existing),
                          faces_rejected=rejected)
