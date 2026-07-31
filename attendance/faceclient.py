"""HTTP client for the separate faceapi recognition service.

The attendance system never loads a model itself — all face work is delegated
over HTTP, so the two services scale and deploy independently.
"""
from __future__ import annotations

import base64

import httpx

from attendance.config import Settings


class FaceServiceError(Exception):
    """faceapi was unreachable or returned an error."""


class FaceClient:
    def __init__(self, settings: Settings) -> None:
        self.base = settings.faceapi_url.rstrip("/")
        self.timeout = settings.faceapi_timeout
        self.headers = {"X-API-Key": settings.faceapi_key} if settings.faceapi_key else {}

    async def _post_json(self, path: str, payload: dict) -> dict:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as c:
                r = await c.post(f"{self.base}{path}", json=payload, headers=self.headers)
        except httpx.HTTPError as e:
            raise FaceServiceError(f"face service unreachable: {e}") from e
        if r.status_code >= 400:
            detail = r.text
            try:
                detail = r.json().get("error", {}).get("message", detail)
            except Exception:
                pass
            raise FaceServiceError(f"face service error {r.status_code}: {detail}")
        return r.json()

    @staticmethod
    def _b64(data: bytes) -> str:
        return base64.b64encode(data).decode()

    # ── operations ───────────────────────────────────────────────────────────
    async def enroll(self, subject_id: str, name: str, images: list[bytes]) -> dict:
        return await self._post_json("/v1/enroll", {
            "subject_id": subject_id, "name": name,
            "images": [self._b64(b) for b in images]})

    async def identify(self, image: bytes) -> dict:
        return await self._post_json("/v1/identify", {"image": self._b64(image)})

    async def verify(self, subject_id: str, image: bytes) -> dict:
        return await self._post_json("/v1/verify",
                                     {"subject_id": subject_id, "image": self._b64(image)})

    async def delete_subject(self, subject_id: str) -> None:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as c:
                await c.delete(f"{self.base}/subjects/{subject_id}", headers=self.headers)
        except httpx.HTTPError:
            pass   # best-effort cleanup

    async def health(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"{self.base}/health", headers=self.headers)
                return r.json() if r.status_code == 200 else {"status": "error"}
        except httpx.HTTPError as e:
            return {"status": "unreachable", "detail": str(e)}
