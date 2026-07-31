"""Attendance system — application factory and wiring.

Domain logic lives in ``services/``, HTTP routes in ``routers/``, persistence in
``db.py`` and all face recognition is delegated over HTTP to the separate
``faceapi`` service via ``faceclient.py``. This module only assembles them.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from attendance.auth import hash_password
from attendance.config import get_settings
from attendance.db import Database
from attendance.faceclient import FaceClient
from attendance.routers import (attendance as attendance_routes, auth, courses,
                                face, reports, sessions, system)

WEB_DIR = Path(__file__).resolve().parent / "web"

API_ROUTERS = (auth.router, face.router, courses.router, sessions.router,
               attendance_routes.router, reports.router, system.router)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.db = Database(settings.db_path)
    app.state.face = FaceClient(settings)
    _bootstrap_admin(app.state.db, settings)
    yield
    app.state.db.close()


def _bootstrap_admin(db: Database, settings) -> None:
    """Create an initial admin so a fresh deployment is usable."""
    if db.one("SELECT 1 FROM users LIMIT 1"):
        return
    db.create_user(settings.bootstrap_admin_email,
                   hash_password(settings.bootstrap_admin_password),
                   "System Administrator", "admin")
    print(f"[attendance] bootstrap admin: {settings.bootstrap_admin_email} "
          f"/ {settings.bootstrap_admin_password}")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Attendance System", version="1.0", lifespan=lifespan,
        description="Face-recognition attendance built on the faceapi service.")

    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_list(),
                       allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code,
                            content={"error": {"code": exc.status_code,
                                               "message": exc.detail}})

    for router in API_ROUTERS:
        app.include_router(router, prefix="/api")

    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    """Serve the PWA. Mounted last so /api/* always takes precedence."""
    if not WEB_DIR.exists():
        return

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(WEB_DIR / "index.html", media_type="text/html")

    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


app = create_app()
