"""Shared route helpers: session isolation, file/analysis lookup, signal loading."""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from fastapi import Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from dsp import utils as utils_mod
from dsp import waterfall as wf_mod

from ..config import settings
from ..models import AnalysisSession, UploadedFile
from ..security import optional_user
from ..storage import cache_read, effective_fs, load_samples


def jsonable(obj: Any) -> Any:
    """Make a DSP result JSON-safe (numpy scalars, NaN, nested structures)."""
    return utils_mod.to_jsonable(obj)


def session_key(x_session_id: str | None = Header(default=None, alias="x-session-id"),
                user=Depends(optional_user)) -> str:
    """Identity of the caller's workspace.

    A signed-in account owns one workspace (``u<user id>``).  Without a token the ``x-session-id``
    header (or ``public``) is used, so the anonymous demo still works while never mixing data
    between visitors.
    """
    if user is not None:
        return f"u{user.id}"
    key = (x_session_id or "").strip()
    if not key:
        return "public"
    key = "".join(ch for ch in key if ch.isalnum() or ch in "-_")[:64]
    return key or "public"


def require_file(db: Session, file_id: str, key: str) -> UploadedFile:
    row = db.get(UploadedFile, file_id)
    if row is None or row.session_key != key:
        raise HTTPException(status_code=404, detail=f"file '{file_id}' does not exist in this session")
    if not os.path.exists(row.stored_path or ""):
        raise HTTPException(status_code=410, detail="the stored file is no longer on disk")
    return row


def require_analysis(db: Session, analysis_id: str, key: str) -> AnalysisSession:
    row = db.get(AnalysisSession, analysis_id)
    if row is None or row.session_key != key:
        raise HTTPException(status_code=404, detail=f"analysis '{analysis_id}' does not exist in this session")
    return row


def analysis_result(row: AnalysisSession) -> dict:
    """Full cached result of an analysis (raises 409 when the analysis did not finish)."""
    if row.status == "running" or row.status == "queued":
        raise HTTPException(status_code=409, detail="the analysis is still running")
    res = cache_read(row.cache_path) if row.cache_path else None
    if res is None:
        if row.status == "failed":
            raise HTTPException(status_code=409, detail=row.error or "the analysis failed")
        raise HTTPException(status_code=410, detail="the cached result of this analysis is gone")
    return res


def file_payload(row: UploadedFile, detail: bool = False) -> dict:
    detection = ((row.meta or {}).get("detection") or {})
    out = {
        "file_id": row.id, "filename": row.filename, "size_bytes": row.size_bytes,
        "sha256": row.sha256, "source": row.source,
        "container": row.container, "format": row.detected_format, "dtype": row.dtype,
        "endianness": row.endianness, "iq_layout": row.iq_layout, "channels": row.channels,
        "n_samples": row.n_samples, "duration_s": row.duration_s,
        "sample_rate": row.sample_rate, "sample_rate_source": row.sample_rate_source,
        "center_frequency": row.center_frequency,
        "center_frequency_source": row.center_frequency_source,
        "confidence": row.confidence, "metadata_source": row.metadata_source,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        # never invent metadata: an absent sample rate is reported as unknown
        "sample_rate_known": bool(row.sample_rate),
        "center_frequency_known": bool(row.center_frequency),
    }
    if detail:
        out["detection"] = jsonable(detection)
        out["detection_evidence"] = jsonable(detection.get("evidence") or detection.get("report") or [])
        out["load_notes"] = jsonable(detection.get("load_notes") or [])
        out["warnings"] = jsonable(detection.get("warnings") or [])
        out["candidates"] = jsonable(detection.get("candidates") or [])
    return out


def prepare_samples(file_row: UploadedFile, options: dict | None = None) -> dict:
    """Load (and optionally band/time-slice) the samples a module should work on.

    ``options`` may carry ``format_override``, ``region`` (``f_lo_hz``/``f_hi_hz``/``t0_s``/``t1_s``)
    and ``max_samples``.  Band extraction converts to a complex baseband record at a decimated rate;
    the crop keeps the requested part of the record.
    """
    opt = dict(options or {})
    loaded = load_samples(file_row, override=opt.get("format_override"),
                          max_samples=opt.get("max_samples"))
    fs, fs_known, fs_source = effective_fs(file_row, loaded, opt.get("format_override"))
    x = np.asarray(loaded.get("samples"))
    notes = list(loaded.get("load_notes") or [])
    region = opt.get("region") or {}
    t0, t1 = region.get("t0_s"), region.get("t1_s")
    f_lo, f_hi = region.get("f_lo_hz"), region.get("f_hi_hz")
    out = {"samples": x, "fs": float(fs), "fs_known": bool(fs_known), "fs_source": fs_source,
           "is_complex": bool(loaded.get("is_complex")), "spec": loaded.get("spec") or {},
           "notes": notes, "region": None, "decimated": False}
    if x.size and (t0 is not None or t1 is not None):
        sl = wf_mod.slice_time(x, fs, t0, t1)
        if sl.get("ok"):
            x = sl["samples"]
            out["time_slice"] = {"i0": int(sl["i0"]), "i1": int(sl["i1"])}
    if x.size and f_lo is not None and f_hi is not None:
        band = wf_mod.extract_band(x, fs, float(f_lo), float(f_hi),
                                   guard_frac=float(opt.get("guard_frac", 0.35)))
        if not band.get("ok"):
            raise HTTPException(status_code=400, detail=f"band extraction failed: {band.get('message')}")
        x = band["samples"]
        out["fs"] = float(band["fs"])
        out["fs_in"] = float(band["fs_in"])
        out["decimated"] = float(band["fs"]) < float(band["fs_in"])
        out["region"] = {"f_lo_hz": float(f_lo), "f_hi_hz": float(f_hi),
                         "bw_hz": float(f_hi) - float(f_lo)}
        out["notes"].append(f"analysed band {float(f_lo):.0f}..{float(f_hi):.0f} Hz shifted to baseband")
    out["samples"] = np.asarray(x)
    out["n_samples"] = int(out["samples"].size)
    return out


def bits_from_analysis(result: dict) -> dict:
    """Pull the demodulated bits / soft values out of a cached analysis result.

    The pipeline stores the demodulator's raw output under the private ``_streams`` key of the
    segment result (it is stripped from browser payloads by ``views.api_view``); a stored stream
    means the module endpoints analyse exactly the bits the pipeline analysed.
    """
    sig = result.get("signal") or {}
    dm = sig.get("demodulation") or {}
    streams = sig.get("_streams") or {}
    bits = streams.get("bits", dm.get("bits"))
    llrs = streams.get("llrs", dm.get("llrs"))
    arr = np.asarray(bits, dtype=np.int8) if bits is not None else None
    soft = np.asarray(llrs, dtype=float) if llrs is not None else None
    if arr is None and soft is not None:
        arr = (soft < 0).astype(np.int8)
    # decoded information bits: the leading FEC hypothesis carries the Viterbi/RS output of the
    # stream it decoded, so the payload-level view (bytes, ASCII runs, header search) can use the
    # bits *after* error correction instead of the raw demodulator decisions
    decoded, decoded_from = None, None
    fec = sig.get("fec") or {}
    best = fec.get("best") or (fec.get("hypotheses") or [None])[0] or {}
    metrics = best.get("metrics") or {}
    info = metrics.get("information_bits") or metrics.get("decoded")
    if isinstance(info, (list, tuple)) and len(info) >= 64 and best.get("confidence", 0) >= 0.2:
        decoded = np.asarray(info, dtype=np.int8).ravel()
        decoded_from = (f"{best.get('hypothesis')} (confidence "
                        f"{float(best.get('confidence') or 0):.2f})")
    return {
        "bits": arr, "soft": soft, "decoded": decoded, "decoded_from": decoded_from,
        "modulation": (streams.get("modulation") or dm.get("modulation")
                       or (sig.get("modulation") or {}).get("primary")),
        "symbol_rate_hz": (streams.get("symbol_rate_hz") or dm.get("symbol_rate_used_hz")
                           or sig.get("symbol_rate_hz")),
        "symbols_i": streams.get("symbols_i"), "symbols_q": streams.get("symbols_q"),
        "source": f"analysis {result.get('_analysis_id')}",
    }
