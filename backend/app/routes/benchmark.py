"""Benchmark mode: the platform tests itself and publishes the confusion matrix."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from dsp import benchmark as bench_mod

from ..db import get_db, session_scope
from ..jobs import Job, job_manager
from ..models import BenchmarkResult
from ..services import run_benchmark
from .common import jsonable, session_key

router = APIRouter(tags=["benchmark"])


@router.get("/benchmark/config", summary="Benchmark grid defaults and available sweep values")
def config() -> dict:
    return {"defaults": jsonable(bench_mod.DEFAULT_CONFIG),
            "kinds": ["amc", "estimators", "demod", "fec", "interleaving", "full"],
            "note": "every case is generated with a known ground truth and analysed by the same "
                    "pipeline the UI uses; the tables below are measured, not declared"}


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    config: dict[str, Any] = Field(default_factory=dict)
    async_job: bool = Field(default=True, alias="async")


def _bench_job(key: str, cfg: dict):
    def fn(job: Job) -> dict:
        with session_scope() as db:
            row = run_benchmark(db, key, cfg, job=job)
            return {"benchmark_id": row.id, "kind": row.kind, "accuracy": row.accuracy,
                    "n_cases": row.n_cases, "duration_ms": row.duration_ms,
                    "status": row.status, "result_url": f"/api/benchmarks/{row.id}"}
    return fn


@router.post("/benchmark", summary="Run the self-benchmark grid in the background")
def run(req: BenchmarkRequest, key: str = Depends(session_key), db: Session = Depends(get_db)) -> dict:
    cfg = dict(req.config or {})
    kind = str(cfg.get("kind") or "full")
    if kind not in ("amc", "estimators", "demod", "fec", "interleaving", "full"):
        raise HTTPException(status_code=422, detail=f"unknown benchmark kind '{kind}'")
    cfg["kind"] = kind
    job = job_manager.submit("benchmark", _bench_job(key, cfg), params={"kind": kind})
    return {"job": job.as_dict(), "config": cfg}


@router.get("/benchmarks", summary="Stored benchmark runs")
def list_benchmarks(limit: int = Query(20, ge=1, le=100), key: str = Depends(session_key),
                    db: Session = Depends(get_db)) -> dict:
    rows = db.scalars(select(BenchmarkResult).where(BenchmarkResult.session_key == key)
                      .order_by(BenchmarkResult.created_at.desc()).limit(limit)).all()
    return {"benchmarks": [{"benchmark_id": r.id, "kind": r.kind, "status": r.status,
                            "accuracy": r.accuracy, "n_cases": r.n_cases,
                            "duration_ms": r.duration_ms, "config": r.config,
                            "created_at": r.created_at.isoformat() if r.created_at else None}
                           for r in rows]}


@router.get("/benchmarks/{benchmark_id}", summary="One benchmark run: confusion matrix and per-class")
def get_benchmark(benchmark_id: str, key: str = Depends(session_key),
                  db: Session = Depends(get_db)) -> dict:
    row = db.get(BenchmarkResult, benchmark_id)
    if row is None or row.session_key != key:
        raise HTTPException(status_code=404, detail="benchmark not found in this session")
    return {"benchmark_id": row.id, "kind": row.kind, "status": row.status,
            "accuracy": row.accuracy, "n_cases": row.n_cases, "duration_ms": row.duration_ms,
            "classes": row.classes, "confusion_matrix": row.confusion_matrix,
            "per_class": row.per_class, "metrics": row.metrics, "config": row.config,
            "notes": row.notes}
