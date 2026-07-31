"""Face Recognition API — application factory and wiring.

Layering
    engine.py      detection / alignment / embedding (no HTTP)
    store.py       subject gallery persistence
    matching.py    similarity scoring
    services.py    recognition operations shared by all transports
    routers/       HTTP surface (multipart, JSON, gallery, system)
    api.py         assembles the above; contains no business logic

Two interchangeable transports are exposed so any client works without an SDK:
multipart/form-data for browsers and curl, JSON+base64 (``/v1/*``) for mobile
and web. Auth is an optional X-API-Key; CORS is open by default.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from faceapi.config import get_settings
from faceapi.engine import FaceEngine
from faceapi.middleware import observability
from faceapi.routers import recognition, recognition_json, subjects, system
from faceapi.store import SubjectStore

WEB_DIR = Path(__file__).resolve().parent / "web"

ROUTERS = (recognition.router, recognition_json.router, subjects.router, system.router)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.engine = FaceEngine(
        detection_model=settings.detection_model_path,
        recognition_model=settings.recognition_model(),
        detection_score_threshold=settings.detection_score_threshold,
        intra_op_threads=settings.intra_op_threads,
        enable_tta=settings.enable_tta,
        detect_max_side=settings.detect_max_side)
    app.state.store = SubjectStore(settings.db_path)
    yield
    try:
        app.state.store.close()
    except Exception:
        pass


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Face Recognition API", version="1.0", lifespan=lifespan,
        description="Cross-platform face enrolment / verification / identification.")

    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()] or ["*"]
    app.add_middleware(CORSMiddleware, allow_origins=origins,
                       allow_methods=["*"], allow_headers=["*"])
    app.middleware("http")(observability)

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code,
                            content={"error": {"code": exc.status_code,
                                               "message": exc.detail,
                                               "path": request.url.path}})

    for router in ROUTERS:
        app.include_router(router)

    _mount_site(app)
    return app


def _mount_site(app: FastAPI) -> None:
    """Product website at /, interactive camera demo at /demo."""

    def _file(name: str, media_type: str):
        path = WEB_DIR / name
        if not path.exists():
            raise HTTPException(404, f"{name} not found")
        return FileResponse(path, media_type=media_type)

    @app.get("/", include_in_schema=False)
    async def product_site():
        if (WEB_DIR / "index.html").exists():
            return _file("index.html", "text/html")
        return JSONResponse({"service": "Face Recognition API", "docs": "/docs"})

    @app.get("/demo", include_in_schema=False)
    async def demo():
        return _file("demo.html", "text/html")

    @app.get("/css/{filename}", include_in_schema=False)
    async def css(filename: str):
        return _file(f"css/{filename}", "text/css")

    @app.get("/js/{filename}", include_in_schema=False)
    async def js(filename: str):
        return _file(f"js/{filename}", "application/javascript")

    @app.get("/icon.svg", include_in_schema=False)
    async def icon():
        return _file("icon.svg", "image/svg+xml")


app = create_app()
