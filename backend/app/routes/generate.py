"""Synthetic signal generator endpoints (real modulations, downloadable IQ/WAV)."""
from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from dsp import synth as synth_mod

from ..config import settings
from ..db import get_db
from ..models import GeneratedSignal, UploadedFile
from ..services import generate_signal
from .common import file_payload, jsonable, require_file, session_key

router = APIRouter(tags=["generator"])

#: the modulations the generator actually implements (kept in sync with dsp.synth's dispatch)
GENERATOR_MODULATIONS = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "2FSK", "GFSK", "AM", "FM"]


@router.get("/generate/options", summary="Generator parameter space (defaults and legal ranges)")
def options() -> dict:
    return {
        "modulations": GENERATOR_MODULATIONS,
        "defaults": jsonable(synth_mod.DEFAULT_SPEC),
        "fec": sorted(synth_mod.FEC_PRESETS.keys()),
        "interleavers": [{"kind": k, **p} for k, p in synth_mod.INTERLEAVER_CANDIDATES],
        "formats": [{"id": "iq", "label": ".iq complex int16 interleaved"},
                    {"id": "wav", "label": ".wav stereo int16 (I left / Q right)"}],
        "payload_kinds": ["random", "text"],
        "ranges": {"snr_db": [-10, 60], "symbol_rate": [100, 5e6], "fs": [1000, 20e6],
                   "n_symbols": [64, 200000], "rolloff": [0.05, 1.0], "mod_index": [0.1, 2.0]},
        "note": "the generated signal is deterministic for a given seed; the returned ground truth "
                "is the generator's own record of what it transmitted",
    }


class GenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    spec: dict[str, Any] = Field(..., description="see GET /api/generate/options")
    format: str = Field(default="iq", description="iq | wav")


@router.post("/generate", summary="Generate a synthetic IQ/WAV file and register it as an input")
def generate(req: GenerateRequest, key: str = Depends(session_key),
             db: Session = Depends(get_db)) -> dict:
    spec = dict(req.spec or {})
    mod = str(spec.get("modulation", "")).upper()
    if mod not in GENERATOR_MODULATIONS:
        raise HTTPException(status_code=422, detail=f"unsupported modulation '{mod}' "
                                                    f"(use one of {', '.join(GENERATOR_MODULATIONS)})")
    spec["modulation"] = mod
    fs = float(spec.get("fs") or 0)
    if fs <= 0:
        raise HTTPException(status_code=422, detail="fs (sample rate) must be a positive number")
    n_sym = int(spec.get("n_symbols") or 0)
    if n_sym < 8:
        raise HTTPException(status_code=422, detail="n_symbols must be at least 8")
    n_samples = int(n_sym * max(2, int(round(fs / float(spec.get("symbol_rate") or fs / 8.0)))))
    if n_samples > settings.max_samples_analyse * 2:
        raise HTTPException(status_code=422,
                            detail=f"that configuration would create {n_samples:,} samples, more "
                                   f"than the {settings.max_samples_analyse * 2:,} sample limit")
    try:
        file_row, gen_row, spec_det = generate_signal(db, key, spec, fmt=req.format)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"the generator rejected a parameter: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "file": file_payload(file_row, detail=True),
        "generated_id": gen_row.id,
        "spec": jsonable(gen_row.spec),
        "ground_truth": jsonable(gen_row.ground_truth),
        "download_url": f"/api/generated/{gen_row.id}/download",
        "analyze_url": f"/api/analyze",
        "note": "the ground truth is what the generator transmitted - compare it with the analysis "
                "output to see the estimator errors",
    }


@router.post("/generator", summary="Alias of POST /api/generate (synthetic signal lab)")
def generator_alias(req: GenerateRequest, key: str = Depends(session_key),
                    db: Session = Depends(get_db)) -> dict:
    return generate(req, key=key, db=db)


@router.post("/generator/recover", summary="Alias of /api/generator/verify (message recovery)")
def recover_alias(payload: dict, key: str = Depends(session_key), db: Session = Depends(get_db)) -> dict:
    from .dspops import VerifyRequest, verify_generated
    return verify_generated(VerifyRequest(**payload), key=key, db=db)


@router.get("/generated", summary="Signals generated in this session")
def generated_list(limit: int = 50, key: str = Depends(session_key),
                   db: Session = Depends(get_db)) -> dict:
    rows = db.scalars(select(GeneratedSignal).where(GeneratedSignal.session_key == key)
                      .order_by(GeneratedSignal.created_at.desc()).limit(limit)).all()
    out = []
    for r in rows:
        f = db.get(UploadedFile, r.file_id) if r.file_id else None
        out.append({"generated_id": r.id, "file_id": r.file_id,
                    "filename": f.filename if f else None,
                    "spec": r.spec, "ground_truth": r.ground_truth, "n_samples": r.n_samples,
                    "duration_s": r.duration_s,
                    "download_url": f"/api/generated/{r.id}/download",
                    "created_at": r.created_at.isoformat() if r.created_at else None})
    return {"generated": out, "count": len(out)}


@router.get("/generated/{generated_id}/download", summary="Download a generated signal file")
def download(generated_id: str, key: str = Depends(session_key), db: Session = Depends(get_db)):
    row = db.get(GeneratedSignal, generated_id)
    if row is None or row.session_key != key:
        raise HTTPException(status_code=404, detail="generated signal not found in this session")
    if not row.file_id:
        raise HTTPException(status_code=410, detail="the generated file has been deleted")
    f = require_file(db, row.file_id, key)
    name = os.path.basename(f.filename or "signal.iq")
    media = "audio/wav" if name.lower().endswith(".wav") else "application/octet-stream"
    return FileResponse(f.stored_path, filename=name, media_type=media)
