"""FastAPI application: API routes, background workers and the built front-end.

Run locally:
    uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import settings
from .db import init_db
from .jobs import job_manager
from .routes import api_router
from .routes import docs as docs_mod

DESCRIPTION = """
**SIH26147 - Automated Model for Analysis of .IQ and .WAV Files Along with Signal Parameter Extraction**

A working analysis platform, not a mock-up: upload a capture (or generate one), and the backend runs the
real DSP chain - format detection, preprocessing, Welch PSD, STFT waterfall, multi-emission detection,
parameter estimation, hybrid modulation classification, blind demodulation, FEC and interleaver
hypothesis testing, bitstream analysis, correlation search and report generation.

Conventions used by every endpoint:

* ``x-session-id`` header isolates your files, analyses and reports from other sessions.
* measured values are returned with ``confidence`` **and** ``method``/``evidence``; when a quantity
  cannot be estimated the record is ``status="unable"`` with ``value=null`` (never a made-up number).
* modulation, FEC, interleaving and protocol statements are *hypotheses* with confidences.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    settings.ensure_dirs()
    docs_mod.sync_index()          # docs/index.json is derived state, refreshed on every start
    job_manager.start()
    app.state.started_at = time.time()
    yield
    job_manager.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name, version=settings.version, description=DESCRIPTION,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "system", "description": "Health, capabilities, statistics and session cleanup."},
            {"name": "files", "description": "Upload, format detection, previews, demo signals."},
            {"name": "analysis", "description": "Run and read back the full analysis pipeline."},
            {"name": "modules", "description": "Individual DSP modules: demod, FEC, interleaving, correlate, compare."},
            {"name": "generator", "description": "Synthetic signal generator with ground truth."},
            {"name": "benchmark", "description": "Self-benchmarking and confusion matrices."},
            {"name": "reports", "description": "PDF / JSON / CSV / TXT exports."},
        ],
    )
    origins = list(settings.cors_origins)
    if "https://rf-signal-intelligence.vercel.app" not in origins:
        origins.append("https://rf-signal-intelligence.vercel.app")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins if origins else ["*"],
        allow_origin_regex=r"^https://rf-signal-intelligence.*\.vercel\.app$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={
            "detail": "the request was rejected by validation (nothing was run)",
            "errors": [{"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")}
                       for e in exc.errors()][:12]})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):            # pragma: no cover
        return JSONResponse(status_code=500, content={
            "detail": f"unhandled {type(exc).__name__}: {exc}",
            "path": str(request.url.path),
            "hint": "the failure was caught by the API boundary: no partial state is reported as a result"})

    @app.get("/api", include_in_schema=False)
    def api_root() -> dict:
        return {"service": settings.app_name, "version": settings.version,
                "problem_statement": settings.problem_statement,
                "docs": "/docs", "openapi": "/openapi.json", "health": "/api/health",
                "documentation": settings.docs_url_path, "docs_api": "/api/docs"}

    app.include_router(api_router, prefix=settings.api_prefix)

    # The repository documentation is served verbatim (Swagger UI keeps /docs, the markdown lives
    # under /documentation, and /api/docs returns the same files as JSON).
    docs_dir = docs_mod.docs_root()
    if docs_dir.is_dir():
        app.mount(settings.docs_url_path, StaticFiles(directory=str(docs_dir), html=False),
                  name="documentation")

    dist = settings.frontend_dist
    if os.path.isdir(dist) and os.path.exists(os.path.join(dist, "index.html")):
        class SPAStaticFiles(StaticFiles):
            """Static files with a single-page-application fallback.

            The UI uses client-side routing, so a deep link such as /dashboard or /analyze/abc is not
            a file on disk.  Instead of a 404 the built index.html is returned and the router takes
            over - which is what makes a refresh (or a restart of the API) harmless.
            """

            async def get_response(self, path, scope):
                try:
                    response = await super().get_response(path, scope)
                except StarletteHTTPException as exc:
                    # Starlette *raises* for a missing file, it does not return a 404 response, so
                    # the fallback has to be implemented here (API/documentation paths excepted).
                    if exc.status_code != 404 or path.startswith(("api/", "documentation/", "docs",
                                                                   "redoc", "openapi.json")):
                        raise
                    response = await super().get_response("index.html", scope)
                # Vite fingerprints the asset names, so hashed files may be cached forever while
                # index.html must always be revalidated - otherwise a rebuilt UI would be loaded
                # with stale CSS/JS after an update (the "CSS loss after restart" failure mode).
                ctype = str(response.headers.get("content-type") or "")
                response.headers["Cache-Control"] = ("no-cache, must-revalidate" if "text/html" in ctype
                                                    else "public, max-age=31536000, immutable")
                return response

        app.mount("/", SPAStaticFiles(directory=dist, html=True), name="frontend")
    else:
        @app.get("/", include_in_schema=False)
        def no_frontend() -> dict:
            return {"message": "the API is running but no front-end build was found",
                    "build_it_with": "cd frontend && npm install && npm run build",
                    "api_docs": "/docs"}
    return app


app = create_app()
