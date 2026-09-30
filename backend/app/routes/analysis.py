"""Analysis endpoints: run the pipeline (async job or inline), read back every artefact."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from dsp import bitstream as bits_mod
from dsp import correlate as corr_mod
from dsp import utils as utils_mod

from ..db import get_db, session_scope
from ..jobs import Job, job_manager
from ..models import AnalysisSession, SignalSegment, UploadedFile
from ..services import run_analysis
from ..storage import cache_read
from ..views import api_view, trim
from .common import (analysis_result, bits_from_analysis, file_payload, jsonable,
                     require_analysis, require_file, session_key)

router = APIRouter(tags=["analysis"])


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    file_id: str
    options: dict[str, Any] | None = None
    kind: str = Field(default="standard", description="standard | blind | region | manual")
    async_job: bool = Field(default=True, alias="async",
                            description="run in a background worker and return a job id")
    wait_s: float = Field(default=0.0, ge=0.0, le=300.0,
                          description="block up to this many seconds waiting for the result")


def _options_for(db: Session, file_row: UploadedFile, raw: dict | None, kind: str) -> dict:
    opt = dict(raw or {})
    det = (file_row.meta or {}).get("detection") or {}
    # a WAV header sample rate is real metadata; an unknown rate must be estimated by the pipeline
    if not opt.get("format_override") and not file_row.sample_rate and det.get("sample_rate"):
        opt["format_override"] = {"sample_rate": det["sample_rate"]}
    opt["blind"] = bool(opt.get("blind")) or kind == "blind"
    if "region" in opt and not isinstance(opt["region"], dict):
        raise HTTPException(status_code=422, detail="options.region must be an object")
    return opt


def _analysis_job(session_id: str, file_id: str, options: dict, kind: str):
    def fn(job: Job) -> dict:
        with session_scope() as db:
            file_row = db.get(UploadedFile, file_id)
            if file_row is None:
                raise ValueError("the file was deleted before the analysis started")
            row = run_analysis(db, session_id, file_row, options, kind=kind, job=job)
            return {"analysis_id": row.id, "status": row.status, "summary": row.summary,
                    "error": row.error, "duration_ms": row.duration_ms,
                    "result_url": f"/api/analysis/{row.id}"}
    return fn


@router.post("/analyze", summary="Run the full analysis pipeline on an uploaded file")
def analyze(req: AnalyzeRequest, key: str = Depends(session_key),
            db: Session = Depends(get_db)) -> dict:
    file_row = require_file(db, req.file_id, key)
    options = _options_for(db, file_row, req.options, req.kind)
    job = job_manager.submit("analyze", _analysis_job(key, req.file_id, options, req.kind),
                             params={"file_id": req.file_id, "kind": req.kind})
    if req.wait_s > 0:
        job = wait_for_job(job.id, timeout=req.wait_s)
    payload = job.as_dict(include_result=True)
    if job.status == "done" and job.result:
        payload["analysis"] = job.result
    payload["poll_url"] = f"/api/jobs/{job.id}"
    payload["note"] = ("the analysis runs in a background worker; poll /api/jobs/{id} for real "
                       "progress and cancellation")
    return payload


@router.post("/blind-analysis", summary="Blind mode: no modulation, rate, FEC or interleaver input")
def blind_analysis(req: AnalyzeRequest, key: str = Depends(session_key),
                   db: Session = Depends(get_db)) -> dict:
    req.kind = "blind"
    raw = dict(req.options or {})
    for k in ("modulation_hint", "symbol_rate_hint", "region", "signal_id"):
        raw.pop(k, None)
    raw["blind"] = True
    raw["run_fec"] = True
    raw["run_interleaving"] = True
    raw["run_hypotheses"] = True
    req.options = raw
    return analyze(req, key=key, db=db)


def wait_for_job(job_id: str, timeout: float = 60.0) -> Job:
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = job_manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"job '{job_id}' is unknown")
        if job.status in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.25)
    return job_manager.get(job_id)                        # type: ignore[return-value]


@router.get("/jobs", summary="Recent background jobs")
def jobs(limit: int = Query(25, ge=1, le=200)) -> dict:
    return {"jobs": job_manager.list(limit=limit), "active": job_manager.active()}


@router.get("/jobs/{job_id}", summary="Job status, progress and result")
def job_status(job_id: str) -> dict:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job '{job_id}' is unknown (it may have been "
                                                    "evicted from the history)")
    return job.as_dict(include_result=True)


@router.post("/jobs/{job_id}/cancel", summary="Ask a running job to stop at the next stage boundary")
def cancel_job(job_id: str) -> dict:
    job = job_manager.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job '{job_id}' is unknown")
    return {"ok": True, "job": job.as_dict()}


# ---------------------------------------------------------------------------------------
# reading results back
# ---------------------------------------------------------------------------------------
def _row_payload(row: AnalysisSession, db: Session) -> dict:
    f = db.get(UploadedFile, row.file_id) if row.file_id else None
    return {
        "analysis_id": row.id, "file_id": row.file_id, "kind": row.kind, "mode": row.mode,
        "status": row.status, "progress": row.progress, "stage_message": row.stage_message,
        "options": row.options, "summary": row.summary, "notes": row.notes, "error": row.error,
        "duration_ms": row.duration_ms,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        "file": file_payload(f) if f else None,
    }


@router.get("/analyze/{analysis_id}/status", summary="Poll a stored analysis: status and progress")
@router.get("/analysis/{analysis_id}/status", include_in_schema=False)
def analysis_status(analysis_id: str, key: str = Depends(session_key),
                    db: Session = Depends(get_db)) -> dict:
    """Lightweight status probe.

    The UI polls the *job* while an analysis runs; this endpoint answers the same question for a
    stored analysis (also after a page reload, when the job id is no longer known).  ``/analysis/...``
    and ``/analyze/...`` are both accepted for convenience.
    """
    row = require_analysis(db, analysis_id, key)
    res = (cache_read(row.cache_path) or {}) if row.status == "done" else {}
    stage = res.get("stages") if isinstance(res, dict) else None
    return {
        "analysis_id": row.id, "status": row.status, "progress": row.progress,
        "stage_message": row.stage_message, "kind": row.kind, "mode": row.mode,
        "duration_ms": row.duration_ms, "error": row.error,
        "n_stages": (stage or {}).get("n_stages"), "n_stages_failed": (stage or {}).get("n_failed"),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "result_url": f"/api/analysis/{row.id}",
        "note": ("the job endpoints (/api/jobs/{id}) report live progress; this endpoint reports the "
                 "stored row, so it keeps working after the job history has been cleared"),
    }


@router.get("/analyses", summary="Analysis history for this session")
def analyses(file_id: str | None = None, kind: str | None = None, status: str | None = None,
             q: str | None = None, limit: int = Query(50, ge=1, le=500), offset: int = 0,
             key: str = Depends(session_key), db: Session = Depends(get_db)) -> dict:
    stmt = select(AnalysisSession).where(AnalysisSession.session_key == key)
    if file_id:
        stmt = stmt.where(AnalysisSession.file_id == file_id)
    if kind:
        stmt = stmt.where(AnalysisSession.kind == kind)
    if status:
        stmt = stmt.where(AnalysisSession.status == status)
    rows = db.scalars(stmt.order_by(AnalysisSession.created_at.desc())
                      .offset(offset).limit(limit)).all()
    items = [_row_payload(r, db) for r in rows]
    if q:
        needle = q.lower()
        items = [i for i in items if needle in str((i.get("file") or {}).get("filename", "")).lower()]
    files = db.scalars(select(UploadedFile).where(UploadedFile.session_key == key)).all()
    return {"analyses": items, "count": len(items), "total_files": len(files)}


@router.get("/analysis/{analysis_id}", summary="Analysis metadata + trimmed result")
def get_analysis(analysis_id: str, full: bool = Query(False, description="include chart arrays"),
                 key: str = Depends(session_key), db: Session = Depends(get_db)) -> dict:
    row = require_analysis(db, analysis_id, key)
    out = {"analysis": _row_payload(row, db)}
    if row.status == "done":
        res = analysis_result(row)
        res["_analysis_id"] = row.id
        out["result"] = api_view(res, spectrogram=False, max_list=6000 if full else 1200)
    return out


@router.get("/analysis/{analysis_id}/result", summary="Full trimmed result (chart arrays included)")
def get_result(analysis_id: str, key: str = Depends(session_key),
               db: Session = Depends(get_db)) -> dict:
    row = require_analysis(db, analysis_id, key)
    res = analysis_result(row)
    res["_analysis_id"] = row.id
    return {"analysis": _row_payload(row, db), "result": api_view(res, spectrogram=False,
                                                                  max_list=8000)}


@router.get("/analysis/{analysis_id}/spectrum", summary="Averaged spectrum / PSD arrays")
def get_spectrum(analysis_id: str, key: str = Depends(session_key),
                 db: Session = Depends(get_db)) -> dict:
    row = require_analysis(db, analysis_id, key)
    res = analysis_result(row)
    sig = res.get("signal") or {}
    rec = res.get("record_spectrum") or {}
    return {"analysis_id": row.id, "fs": res.get("fs"),
            "record_spectrum": trim(rec, max_list=8000),
            "segment_spectrum": trim(sig.get("spectrum") or {}, max_list=8000),
            "segment_center_hz": res.get("segment_center_hz")}


@router.get("/analysis/{analysis_id}/spectrogram", summary="Waterfall matrix (rounded, downsampled)")
def get_spectrogram(analysis_id: str, max_freq: int = Query(0, ge=0, le=2048),
                    max_time: int = Query(0, ge=0, le=4096), key: str = Depends(session_key),
                    db: Session = Depends(get_db)) -> dict:
    row = require_analysis(db, analysis_id, key)
    res = analysis_result(row)
    pw = res.get("spectrogram") or {}
    if not pw or not pw.get("db"):
        return {"analysis_id": row.id, "ok": False,
                "message": pw.get("message") or "no spectrogram was computed for this analysis"}
    dbm = np.asarray(pw["db"], dtype=float)
    f = np.asarray(pw.get("freq_hz") or np.arange(dbm.shape[0]), dtype=float)
    t = np.asarray(pw.get("times_s") or np.arange(dbm.shape[1]), dtype=float)
    if max_freq and dbm.shape[0] > max_freq:
        idx = np.linspace(0, dbm.shape[0] - 1, max_freq).astype(int)
        dbm, f = dbm[idx], f[idx]
    if max_time and dbm.shape[1] > max_time:
        idx = np.linspace(0, dbm.shape[1] - 1, max_time).astype(int)
        dbm, t = dbm[:, idx], t[idx]
    return {
        "analysis_id": row.id, "ok": True,
        "freq_hz": np.round(f, 2).tolist(), "times_s": np.round(t, 6).tolist(),
        "db": np.round(dbm, 2).tolist(),
        "shape": [int(dbm.shape[0]), int(dbm.shape[1])],
        "orientation": "db[frequency_index][time_index]",
        "meta": {k: v for k, v in pw.items() if k not in ("db", "freq_hz", "times_s")},
        "detection": trim(res.get("detection") or {}, max_list=200),
        "note": "matrix downsampled for transport; set max_freq/max_time to control the resolution",
    }


@router.get("/analysis/{analysis_id}/detection", summary="Detected emissions with their evidence")
def get_detection(analysis_id: str, key: str = Depends(session_key),
                  db: Session = Depends(get_db)) -> dict:
    row = require_analysis(db, analysis_id, key)
    res = analysis_result(row)
    return {"analysis_id": row.id, "detection": trim(res.get("detection") or {}, max_list=500),
            "selected_signal": trim(res.get("selected_signal") or {}, max_list=500),
            "segmentation": trim(res.get("segmentation") or {}, max_list=500)}


@router.get("/analysis/{analysis_id}/segments", summary="Persisted segment rows")
def get_segments(analysis_id: str, key: str = Depends(session_key),
                 db: Session = Depends(get_db)) -> dict:
    row = require_analysis(db, analysis_id, key)
    rows = db.scalars(select(SignalSegment).where(SignalSegment.analysis_id == row.id)
                      .order_by(SignalSegment.segment_index)).all()
    return {"analysis_id": row.id, "segments": [{
        "segment_id": s.id, "index": s.segment_index, "label": s.label, "kind": s.kind,
        "t0_s": s.t0_s, "t1_s": s.t1_s, "duration_s": s.duration_s,
        "f_lo_hz": s.f_lo_hz, "f_hi_hz": s.f_hi_hz, "center_frequency_hz": s.center_frequency_hz,
        "peak_frequency_hz": s.peak_frequency_hz,
        "bandwidth_hz": s.bandwidth_hz, "peak_power_db": s.peak_power_db,
        "mean_power_db": s.mean_power_db, "snr_db": s.snr_db, "confidence": s.confidence,
        "duty_cycle": s.duty_cycle, "n_bursts": s.n_bursts, "selected": s.selected,
        "evidence": (s.meta or {}).get("evidence"), "limitations": (s.meta or {}).get("limitations"),
        "effective_bandwidth_hz": (s.meta or {}).get("effective_bandwidth_hz"),
    } for s in rows]}


@router.get("/analysis/{analysis_id}/parameters", summary="Every extracted parameter with method")
def get_parameters(analysis_id: str, key: str = Depends(session_key),
                   db: Session = Depends(get_db)) -> dict:
    row = require_analysis(db, analysis_id, key)
    res = analysis_result(row)
    sig = res.get("signal") or {}
    rec = res.get("record_spectrum") or {}
    groups = dict(sig.get("params") or {})
    rec_params = rec.get("params") or {}
    flat: list[dict] = []
    for group, value in (list(groups.items()) + ([("record", rec_params)] if rec_params else [])):
        if isinstance(value, dict):
            for key_name, rec_row in value.items():
                if isinstance(rec_row, dict):
                    flat.append(dict(rec_row, group=group))
        elif isinstance(value, list):
            for rec_row in value:
                if isinstance(rec_row, dict):
                    flat.append(dict(rec_row, group=group))
    return {"analysis_id": row.id, "groups": jsonable(groups),
            "record": jsonable(rec_params), "parameters": jsonable(flat),
            "quality": jsonable(res.get("quality")), "confidence": jsonable(sig.get("confidence"))}


@router.get("/analysis/{analysis_id}/constellation", summary="Constellation view(s) + EVM")
def get_constellation(analysis_id: str, view: str | None = None, key: str = Depends(session_key),
                      db: Session = Depends(get_db)) -> dict:
    row = require_analysis(db, analysis_id, key)
    res = analysis_result(row)
    sig = res.get("signal") or {}
    return {"analysis_id": row.id, "constellation": trim(sig.get("constellation") or {}, max_list=6000),
            "demodulation": trim(sig.get("demodulation") or {}, max_list=20)}


@router.get("/analysis/{analysis_id}/eye", summary="Eye diagram traces and timing metrics")
def get_eye(analysis_id: str, key: str = Depends(session_key), db: Session = Depends(get_db)) -> dict:
    row = require_analysis(db, analysis_id, key)
    res = analysis_result(row)
    sig = res.get("signal") or {}
    # the eye traces are the point of this endpoint: keep them (they are bounded already)
    return {"analysis_id": row.id,
            "eye": trim(sig.get("eye") or {}, max_list=600, drop_heavy=False)}


@router.get("/analysis/{analysis_id}/bitstream", summary="Bit/byte level analysis of the bit stream")
def get_bitstream(analysis_id: str, stream: str = Query("auto", pattern="^(auto|demod|decoded)$"),
                  max_bits: int = Query(262144, ge=64, le=4_000_000),
                  key: str = Depends(session_key), db: Session = Depends(get_db)) -> dict:
    """Byte/bit statistics of the selected stream.

    ``demod`` = the raw decisions of the demodulator, ``decoded`` = the information bits produced by
    the leading FEC hypothesis (error corrected, when one was found), ``auto`` = decoded bits when
    they exist, otherwise the demodulator decisions.  Both views are always described so the numbers
    can never be mistaken for a different stream.
    """
    row = require_analysis(db, analysis_id, key)
    res = analysis_result(row)
    info = bits_from_analysis(res)
    dm = ((res.get("signal") or {}).get("demodulation")) or {}
    have_decoded = info.get("decoded") is not None
    if stream == "decoded" and not have_decoded:
        return {"analysis_id": row.id, "ok": False,
                "message": "no error-corrected (FEC-decoded) bit stream is available for this "
                           "analysis", "hint": "run POST /api/fec/analyze first; the decoded bits "
                                               "are only shown for a hypothesis that passed the "
                                               "0.20 confidence threshold"}
    use_decoded = have_decoded if stream == "auto" else (stream == "decoded")
    bits = info["decoded"] if use_decoded else info.get("bits")
    if bits is None or np.size(bits) < 8:
        return {"analysis_id": row.id, "ok": False,
                "message": "no demodulated bit stream is available for this analysis",
                "hint": "run the demodulator first (the Analyze pipeline does it automatically)"}
    arr = np.asarray(bits, dtype=np.int8).ravel()[:max_bits]
    data = np.packbits(arr[: (arr.size // 8) * 8].astype(np.uint8)).astype(np.uint8)

    def _stats(stream_arr: np.ndarray) -> dict:
        st = np.packbits(stream_arr[: (stream_arr.size // 8) * 8].astype(np.uint8)).astype(np.uint8)
        return {"n_bits": int(stream_arr.size),
                "entropy": {"bits_per_bit": bits_mod.shannon_entropy_bits(stream_arr),
                            "bits_per_byte": bits_mod.shannon_entropy_bytes(st)},
                "ones_fraction": float(np.mean(stream_arr)) if stream_arr.size else None,
                "byte_histogram": bits_mod.byte_histogram(st),
                "printable_ascii": bits_mod.printable_ascii_candidate(st),
                "frame_candidates": bits_mod.frame_boundary_candidates(st, max_period=2048),
                "periodicity": bits_mod.repeat_periodicity(stream_arr, max_period=2048),
                "ngrams": bits_mod.repeated_ngrams(st, n=4, min_count=3, max_report=8)}

    out = {
        "analysis_id": row.id, "ok": True, "stream_used": "decoded" if use_decoded else "demod",
        "stream_note": (f"error-corrected information bits from {info.get('decoded_from')}"
                        if use_decoded else
                        "raw demodulator decisions (no error correction applied)"),
        "decoded_available": bool(have_decoded),
        "decoded_from": info.get("decoded_from"),
        "n_bits": int(arr.size), "n_bits_total": int(np.size(bits)),
        "modulation": info.get("modulation") or dm.get("modulation"),
        "symbol_rate_used_hz": info.get("symbol_rate_hz") or dm.get("symbol_rate_used_hz"),
        "bit_source": {k: v for k, v in info.items()
                       if k in ("modulation", "symbol_rate_hz", "source", "decoded_from")},
        "bits_preview": "".join(str(int(b)) for b in arr[:512]),
        "hex_dump": bits_mod.hex_dump(data, max_bytes=256),
        "statistics": _stats(arr),
        "entropy": {"bits_per_bit": bits_mod.shannon_entropy_bits(arr),
                    "bits_per_byte": bits_mod.shannon_entropy_bytes(data)},
        "byte_histogram": bits_mod.byte_histogram(data),
        "run_length": bits_mod.run_length_stats(arr),
        "autocorrelation": bits_mod.bit_autocorrelation(arr, max_lag=128),
        "periodicity": bits_mod.repeat_periodicity(arr, max_period=2048),
        "ngrams": bits_mod.repeated_ngrams(data, n=4, min_count=3, max_report=12),
        "frame_candidates": bits_mod.frame_boundary_candidates(data, max_period=2048),
        "printable_ascii": bits_mod.printable_ascii_candidate(data),
        "summary": bits_mod.summarise_bits(arr),
        "comparison": ({"demod": _stats(np.asarray(info["bits"], dtype=np.int8).ravel()[:max_bits]),
                        "decoded": _stats(info["decoded"][:max_bits]),
                        "note": "statistics before and after error correction on the same record"}
                       if have_decoded else None),
        "note": "bit and byte statistics of the demodulated stream; a printable-ASCII run is a "
                "candidate reading of the bits, not a decoded message",
    }
    auto = ((res.get("signal") or {}).get("correlation") or {}).get("auto")
    out["repeated_structure"] = jsonable(auto) if auto else None
    return jsonable(out)


@router.get("/analysis/{analysis_id}/evidence", summary="Evidence graph + confidence system")
def get_evidence(analysis_id: str, key: str = Depends(session_key),
                 db: Session = Depends(get_db)) -> dict:
    row = require_analysis(db, analysis_id, key)
    res = analysis_result(row)
    sig = res.get("signal") or {}
    return {"analysis_id": row.id,
            "evidence_graph": trim(sig.get("evidence_graph") or res.get("evidence_graph") or {},
                                   max_list=200),
            "confidence": trim(sig.get("confidence") or {}, max_list=50),
            "quality": trim(res.get("quality") or {}, max_list=50),
            "stages": trim(res.get("stages") or {}, max_list=64),
            "hypotheses": trim(sig.get("hypotheses") or {}, max_list=32),
            "notes": res.get("notes"), "warnings": res.get("warnings")}


class RegionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    f_lo_hz: float | None = None
    f_hi_hz: float | None = None
    t0_s: float | None = None
    t1_s: float | None = None
    options: dict[str, Any] | None = None
    async_job: bool = Field(default=True, alias="async")
    wait_s: float = 0.0


@router.post("/analysis/{analysis_id}/region", summary="Re-analyse a selected time/frequency region")
def reanalyse_region(analysis_id: str, req: RegionRequest, key: str = Depends(session_key),
                     db: Session = Depends(get_db)) -> dict:
    row = require_analysis(db, analysis_id, key)
    if row.file_id is None:
        raise HTTPException(status_code=409, detail="this analysis has no file attached")
    region = {k: v for k, v in (("f_lo_hz", req.f_lo_hz), ("f_hi_hz", req.f_hi_hz),
                                ("t0_s", req.t0_s), ("t1_s", req.t1_s)) if v is not None}
    if not region:
        raise HTTPException(status_code=422, detail="select at least one of f_lo_hz/f_hi_hz/t0_s/t1_s")
    if ("f_lo_hz" in region) != ("f_hi_hz" in region):
        raise HTTPException(status_code=422, detail="f_lo_hz and f_hi_hz must be given together")
    if ("t0_s" in region) != ("t1_s" in region):
        raise HTTPException(status_code=422, detail="t0_s and t1_s must be given together")
    options = dict(req.options or {})
    options["region"] = region
    file_row = require_file(db, row.file_id, key)
    options = _options_for(db, file_row, options, "region")
    job = job_manager.submit("analyze", _analysis_job(key, row.file_id, options, "region"),
                             params={"file_id": row.file_id, "kind": "region", "region": region})
    out = job.as_dict(include_result=True)
    out["region"] = region
    out["parent_analysis_id"] = analysis_id
    return out


class SelectRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    signal_id: int = Field(..., description="index of the detected emission (0 = first)")
    options: dict[str, Any] | None = None
    async_job: bool = Field(default=True, alias="async")
    wait_s: float = 0.0


@router.post("/analysis/{analysis_id}/select-signal",
             summary="Re-analyse one of the detected emissions (click-to-analyze)")
def select_signal(analysis_id: str, req: SelectRequest, key: str = Depends(session_key),
                  db: Session = Depends(get_db)) -> dict:
    row = require_analysis(db, analysis_id, key)
    res = analysis_result(row)
    sigs = ((res.get("detection") or {}).get("signals") or [])
    if not 0 <= req.signal_id < len(sigs):
        raise HTTPException(status_code=422,
                            detail=f"signal_id must be between 0 and {max(0, len(sigs) - 1)} "
                                   f"({len(sigs)} emissions were detected)")
    file_row = require_file(db, row.file_id, key)
    options = dict(req.options or {})
    options["signal_id"] = int(req.signal_id)
    options = _options_for(db, file_row, options, "manual")
    job = job_manager.submit("analyze", _analysis_job(key, row.file_id, options, "manual"),
                             params={"file_id": row.file_id, "signal_id": req.signal_id})
    out = job.as_dict(include_result=True)
    out["signal"] = jsonable(sigs[req.signal_id])
    out["parent_analysis_id"] = analysis_id
    return out


@router.delete("/analysis/{analysis_id}", summary="Delete a stored analysis and its cache")
def delete_analysis(analysis_id: str, key: str = Depends(session_key),
                    db: Session = Depends(get_db)) -> dict:
    row = require_analysis(db, analysis_id, key)
    db.delete(row)
    db.commit()
    return {"ok": True, "analysis_id": analysis_id}
