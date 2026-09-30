"""Health, capabilities, storage statistics and session lifecycle."""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import shutil
import sys
import time

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

import dsp
from dsp import fec as fec_mod
from dsp import preprocess as pre_mod
from dsp import scenarios as scen_mod
from dsp import synth as synth_mod

from ..config import DANGEROUS_EXTENSIONS, settings
from ..db import counts, get_db, init_db
from ..jobs import job_manager
from ..models import AnalysisSession, GeneratedSignal, Report, UploadedFile
from ..storage import delete_file
from .common import session_key

router = APIRouter(tags=["system"])
_START = time.time()


@router.get("/health", summary="Liveness probe with real subsystem checks")
def health(db: Session = Depends(get_db)) -> dict:
    checks: dict[str, dict] = {}
    try:
        init_db()                       # idempotent: the probe must not depend on startup timing
    except Exception as exc:                                            # pragma: no cover
        checks["schema"] = {"ok": False, "error": str(exc)}
    try:
        c = counts(db)
        checks["database"] = {"ok": True, "tables": len(c), "rows": int(sum(c.values()))}
    except Exception as exc:                                            # pragma: no cover
        checks["database"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        settings.ensure_dirs()
        probe = os.path.join(settings.data_dir, ".health")
        with open(probe, "w") as fh:
            fh.write("ok")
        os.remove(probe)
        checks["storage"] = {"ok": True, "data_dir": settings.data_dir}
    except Exception as exc:                                            # pragma: no cover
        checks["storage"] = {"ok": False, "error": str(exc)}
    checks["dsp"] = {"ok": bool(dsp.__version__), "version": dsp.__version__,
                     "modules": len([m for m in dir(dsp) if not m.startswith("_")])}
    checks["jobs"] = {"ok": True, "active": job_manager.active(), "workers": job_manager.workers}
    ok = all(v.get("ok") for v in checks.values())
    return {"status": "ok" if ok else "degraded", "version": settings.version,
            "problem_statement": settings.problem_statement,
            "server_time": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "uptime_s": round(time.time() - _START, 2), "checks": checks}


@router.get("/capabilities", summary="Everything the platform can actually do")
def capabilities() -> dict:
    return {
        "modulations": ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "2FSK", "GFSK", "AM", "FM"],
        "demodulators": ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "2FSK", "GFSK", "AM", "FM"],
        "fec_families": ["convolutional (Viterbi)", "Reed-Solomon", "concatenated", "LDPC (not implemented - reported as unsupported)"],
        "fec_presets": sorted(synth_mod.FEC_PRESETS.keys()),
        "interleavers": [{"kind": k, **p} for k, p in synth_mod.INTERLEAVER_CANDIDATES],
        "dtypes": ["int8", "uint8", "int16", "uint16", "int32", "float32", "float64"],
        "containers": ["wav", "raw/interleaved binary", "npy", "sigmf-meta (sidecar)"],
        "extensions": list(settings.allowed_extensions),
        "refused_extensions": list(DANGEROUS_EXTENSIONS),
        "preprocess_steps": [s.get("id") if isinstance(s, dict) else str(s)
                             for s in (getattr(pre_mod, "DEFAULT_STEPS", []) or [])],
        "scenarios": [row.get("id") for row in (getattr(scen_mod, "SCENARIOS", []) or [])
                      if isinstance(row, dict)],
        "demo_signals": sorted(list(getattr(scen_mod, "DEMO_PRESETS", {}).keys())),
        "limits": {"max_upload_bytes": settings.max_upload_bytes,
                   "max_samples_analyse": settings.max_samples_analyse,
                   "max_samples_display": settings.max_samples_display},
    }


@router.get("/stats", summary="Storage usage and per-table row counts for this session")
def stats(key: str = Depends(session_key), db: Session = Depends(get_db)) -> dict:
    files = db.scalars(select(UploadedFile).where(UploadedFile.session_key == key)).all()
    analyses = db.scalars(select(AnalysisSession).where(AnalysisSession.session_key == key)).all()
    reports = db.scalars(select(Report).where(Report.session_key == key)).all()
    gen = db.scalars(select(GeneratedSignal).where(GeneratedSignal.session_key == key)).all()
    usage = shutil.disk_usage(settings.data_dir)
    by_status: dict[str, int] = {}
    for a in analyses:
        by_status[a.status] = by_status.get(a.status, 0) + 1
    return {
        "session": key,
        "files": len(files), "files_bytes": int(sum(f.size_bytes or 0 for f in files)),
        "analyses": len(analyses), "analyses_by_status": by_status,
        "reports": len(reports), "generated_signals": len(gen),
        "tables": counts(db),
        "disk": {"total_gb": round(usage.total / 1e9, 2), "used_gb": round(usage.used / 1e9, 2),
                 "free_gb": round(usage.free / 1e9, 2)},
        "jobs": {"active": job_manager.active(), "workers": job_manager.workers},
    }


@router.get("/settings", summary="Effective configuration (read-only) + runtime environment")
def get_settings(key: str = Depends(session_key)) -> dict:
    return {
        "read_only": True,
        "note": "These values are deployment configuration (environment variables). They are shown "
                "so the run can be reproduced; they are not silently rewritten by the UI.",
        "app": {"name": settings.app_name, "version": settings.version,
                "problem_statement": settings.problem_statement, "api_prefix": settings.api_prefix},
        "paths": {"data": settings.data_dir, "uploads": settings.upload_dir,
                  "generated": settings.generated_dir, "reports": settings.report_dir,
                  "cache": settings.cache_dir, "sample_data": settings.sample_dir},
        "limits": {"max_upload_bytes": settings.max_upload_bytes,
                   "max_samples_analyse": settings.max_samples_analyse,
                   "max_samples_display": settings.max_samples_display,
                   "job_workers": settings.job_workers, "job_history": settings.job_history},
        "security": {"session_header": settings.session_header,
                     "allowed_extensions": list(settings.allowed_extensions),
                     "refused_extensions": list(DANGEROUS_EXTENSIONS),
                     "uploads_are_never_executed": True},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "numpy": np.__version__},
        "ml": _ml_info(),
    }


def _ml_info() -> dict:
    base = os.path.join(os.path.dirname(settings.data_dir), "ml", "models")
    rep = os.path.join(base, "modclass_report.json")
    info: dict = {"model_dir": base}
    try:
        with open(rep) as fh:
            info["report"] = json.load(fh)
    except Exception:
        info["report"] = None
    info["classifier_present"] = os.path.exists(os.path.join(base, "modclass.joblib"))
    return info


@router.get("/models", summary="Machine-learning model availability and measured scores")
def models() -> dict:
    info = _ml_info()
    rep = info.get("report") or {}
    return {
        "classifier": {"available": info.get("classifier_present"), "kind": "RandomForest (scikit-learn)",
                       "accuracy": rep.get("accuracy"), "n_features": rep.get("n_features"),
                       "trained_on": rep.get("trained_on") or rep.get("config"),
                       "notes": rep.get("notes")},
        "usage": "The hybrid classifier uses DSP rules and statistics as its primary decision path; "
                 "the ML model is an additional vote, never the only dependency.",
    }


@router.delete("/session", summary="Delete every file, analysis and report belonging to this session")
def clear_session(key: str = Depends(session_key), db: Session = Depends(get_db)) -> dict:
    files = db.scalars(select(UploadedFile).where(UploadedFile.session_key == key)).all()
    for f in files:
        for a in db.scalars(select(AnalysisSession).where(AnalysisSession.file_id == f.id)).all():
            db.delete(a)
        delete_file(f)
        db.delete(f)
    removed = len(files)
    for a in db.scalars(select(AnalysisSession).where(AnalysisSession.session_key == key)).all():
        db.delete(a)
    for r in db.scalars(select(Report).where(Report.session_key == key)).all():
        db.delete(r)
    db.commit()
    return {"ok": True, "removed_files": removed, "session": key,
            "message": "session data removed (temporary files and cached results cleaned up)"}
