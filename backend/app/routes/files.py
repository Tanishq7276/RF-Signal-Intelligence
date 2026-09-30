"""Upload, format inspection, preview, download and the shipped demo signals."""
from __future__ import annotations

import os

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from dsp import iqformats as iqf
from dsp import scenarios as scen_mod
from dsp import utils as utils_mod

from ..config import settings
from ..db import get_db
from ..models import AnalysisSession, GeneratedSignal, UploadedFile
from ..services import load_demo_signal
from ..storage import delete_file, file_model_from_spec, save_stream
from .common import file_payload, jsonable, prepare_samples, require_file, session_key

router = APIRouter(tags=["files"])


@router.post("/upload", summary="Upload an .iq/.wav/complex-IQ capture and detect its format")
async def upload(file: UploadFile = File(...), key: str = Depends(session_key),
                 db: Session = Depends(get_db)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="the upload has no filename")
    try:
        spec = save_stream(key, file.filename, file.file, source="upload")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:                                            # pragma: no cover
        raise HTTPException(status_code=500, detail=f"the upload could not be stored: {exc}") from exc
    row = file_model_from_spec(key, spec)
    db.add(row)
    db.commit()
    db.refresh(row)
    det = (row.meta or {}).get("detection") or {}
    return {
        "file": file_payload(row, detail=True),
        "detection_report": jsonable(iqf.file_report(det)),
        "message": "file stored and parsed; metadata that is absent from the file is reported as "
                   "'Unknown / requires estimation' and never invented",
    }


@router.get("/files", summary="List the files of this session")
def list_files(source: str | None = None, limit: int = Query(100, ge=1, le=500),
               key: str = Depends(session_key), db: Session = Depends(get_db)) -> dict:
    q = select(UploadedFile).where(UploadedFile.session_key == key)
    if source:
        q = q.where(UploadedFile.source == source)
    rows = db.scalars(q.order_by(UploadedFile.created_at.desc()).limit(limit)).all()
    return {"files": [file_payload(r) for r in rows], "count": len(rows)}


@router.get("/files/{file_id}", summary="Full detection detail and evidence for one file")
def get_file(file_id: str, key: str = Depends(session_key), db: Session = Depends(get_db)) -> dict:
    row = require_file(db, file_id, key)
    det = (row.meta or {}).get("detection") or {}
    return {"file": file_payload(row, detail=True), "detection_report": jsonable(iqf.file_report(det))}


@router.get("/files/{file_id}/preview", summary="Downsampled time-domain / I-Q preview for the UI")
def preview(file_id: str, max_points: int = Query(4000, ge=128, le=20000),
            key: str = Depends(session_key), db: Session = Depends(get_db)) -> dict:
    row = require_file(db, file_id, key)
    try:
        loaded = prepare_samples(row, {"max_samples": settings.max_samples_display})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"the file could not be loaded: {exc}") from exc
    x = np.asarray(loaded["samples"])
    fs = loaded["fs"]
    if x.size == 0:
        raise HTTPException(status_code=400, detail="the file contains no samples")
    i = np.real(x).astype(float)
    q = np.imag(x).astype(float) if np.iscomplexobj(x) else np.zeros_like(i)
    n_max = max_points
    i_d = utils_mod.decimate_for_plot(i, n_max)
    q_d = utils_mod.decimate_for_plot(q, n_max)
    t = np.linspace(0.0, x.size / fs, i_d.size, endpoint=False) if fs else np.arange(i_d.size) * 1.0
    n_fft = int(min(65536, 2 ** int(np.floor(np.log2(max(1024, x.size))))))
    seg = x[:n_fft] * np.hanning(n_fft)
    span_db = np.abs(np.fft.fftshift(np.fft.fft(seg))) ** 2
    span_db = 10 * np.log10(np.maximum(span_db, 1e-30))
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / fs)) if fs else np.arange(n_fft)
    mag = np.abs(x)
    return {
        "file": file_payload(row),
        "fs": fs, "fs_known": loaded["fs_known"], "fs_source": loaded["fs_source"],
        "n_samples": int(x.size), "is_complex": bool(loaded["is_complex"]),
        "t_s": utils_mod.to_jsonable(t),
        "i": utils_mod.to_jsonable(i_d), "q": utils_mod.to_jsonable(q_d),
        "envelope": utils_mod.to_jsonable(utils_mod.decimate_for_plot(mag, n_max)),
        "phase_deg": utils_mod.to_jsonable(utils_mod.decimate_for_plot(
            np.degrees(np.unwrap(np.angle(x[:min(x.size, 20000)]))) if np.iscomplexobj(x)
            else np.zeros(min(x.size, 20000)), n_max)),
        "fft": {"freq_hz": utils_mod.to_jsonable(utils_mod.decimate_for_plot(freqs, 2000)),
                "db": utils_mod.to_jsonable(utils_mod.decimate_for_plot(span_db, 2000)),
                "n_fft": n_fft, "window": "Hann"},
        "stats": {
            "dc_i": float(np.mean(i)), "dc_q": float(np.mean(q)),
            "rms": float(np.sqrt(np.mean(np.abs(x) ** 2))),
            "peak": float(np.max(mag)) if mag.size else 0.0,
            "papr_db": float(20 * np.log10((np.max(mag) + 1e-12) /
                                           (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12))),
            "n_clipped": int(np.sum(mag >= 0.999 * np.max(mag))) if mag.size else 0,
        },
        "notes": loaded["notes"],
        "downsampled": True,
        "max_points": n_max,
    }


@router.get("/files/{file_id}/download", summary="Download the stored (unmodified) file")
def download(file_id: str, key: str = Depends(session_key), db: Session = Depends(get_db)):
    row = require_file(db, file_id, key)
    return FileResponse(row.stored_path, filename=row.filename or "signal.iq",
                        media_type="application/octet-stream")


@router.delete("/files/{file_id}", summary="Delete a file and every analysis derived from it")
def remove_file(file_id: str, key: str = Depends(session_key), db: Session = Depends(get_db)) -> dict:
    row = require_file(db, file_id, key)
    analyses = db.scalars(select(AnalysisSession).where(AnalysisSession.file_id == row.id)).all()
    for a in analyses:
        db.delete(a)
    gens = db.scalars(select(GeneratedSignal).where(GeneratedSignal.file_id == row.id)).all()
    for g in gens:
        g.file_id = None
    delete_file(row)
    db.delete(row)
    db.commit()
    return {"ok": True, "removed_analyses": len(analyses), "file_id": file_id}


@router.get("/demo", summary="List the shipped demo signals with their (known) specification")
def demo_list(key: str = Depends(session_key), db: Session = Depends(get_db)) -> dict:
    items = []
    registered = {r.filename for r in db.scalars(
        select(UploadedFile).where(UploadedFile.session_key == key)).all()}
    for name, preset in scen_mod.DEMO_PRESETS.items():
        path = os.path.join(settings.sample_dir, name)
        items.append({
            "name": name, "description": preset.get("description"),
            "available_on_disk": os.path.exists(path),
            "size_bytes": os.path.getsize(path) if os.path.exists(path) else None,
            "registered": name in registered,
            "known_parameters": {k: v for k, v in preset.items() if k != "description"},
        })
    return {"demo_signals": items, "count": len(items),
            "note": "the parameters listed here belong to the generator that produced the demo "
                    "files; they are shown as ground truth for the demo, not as analysis output"}


@router.post("/demo/{name}", summary="Register a demo signal (regenerates it if the file is missing)")
def demo_load(name: str, key: str = Depends(session_key), db: Session = Depends(get_db)) -> dict:
    safe = os.path.basename(name)
    if safe not in scen_mod.DEMO_PRESETS:
        raise HTTPException(status_code=404, detail=f"'{safe}' is not one of the shipped demo signals")
    out = load_demo_signal(db, key, safe)
    if out is None:
        raise HTTPException(status_code=500, detail="the demo signal could not be prepared")
    row, info = out
    return {"file": file_payload(row, detail=True), "info": info,
            "path_used": "sample_data" if not info.get("regenerated") else "regenerated",
            "next_step": "POST /api/analyze with this file_id to run the full pipeline"}
