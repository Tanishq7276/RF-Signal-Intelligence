"""API routers.  Every route is thin: validation + service call + JSON serialisation."""
from __future__ import annotations

from fastapi import APIRouter

from . import (analysis, auth, benchmark, docs, dspops, files, generate, modules, reports,
               system)

api_router = APIRouter()
api_router.include_router(system.router)
api_router.include_router(docs.router)
api_router.include_router(auth.router)
api_router.include_router(files.router)
api_router.include_router(analysis.router)
api_router.include_router(modules.router)
api_router.include_router(dspops.router)
api_router.include_router(generate.router)
api_router.include_router(benchmark.router)
api_router.include_router(reports.router)

__all__ = ["api_router"]
