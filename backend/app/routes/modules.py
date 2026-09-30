"""Single-module endpoints: demodulation, FEC, interleaving, correlation and comparison.

Each route works either
  * on an existing analysis (``analysis_id``) - it reuses the exact demodulated bit stream and the
    selected emission of that analysis, or
  * directly on a file (``file_id``) with explicit parameters, optionally limited to a band/time
    region, in which case the module runs on freshly loaded samples.
Nothing is cached between the two paths: the numbers returned are measured for the request.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from dsp import correlate as corr_mod
from dsp import demod as demod_mod
from dsp import fec as fec_mod
from dsp import interleave as int_mod
from dsp import utils as utils_mod

from ..db import get_db
from ..models import (AnalysisSession, Comparison, CorrelationResult, DemodulationResult,
                      FECHypothesis, InterleavingHypothesis, UploadedFile)
from ..services import compare_results
from ..views import trim
from .common import (analysis_result, bits_from_analysis, jsonable, prepare_samples, require_analysis,
                     require_file, session_key)

router = APIRouter(tags=["modules"])


def _num(value):
    """Scalar float or None - the demodulator returns BER/EsN0 inside a nested dict."""
    if isinstance(value, dict):
        for key in ("ber_estimate", "value", "estimate"):
            if isinstance(value.get(key), (int, float)):
                return float(value[key])
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _persist(db: Session, model, payload: dict):
    cols = {c.name for c in model.__table__.columns}
    row = model(**{k: v for k, v in payload.items() if k in cols})
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _resolve(key: str, db: Session, analysis_id: str | None, file_id: str | None):
    """Return (analysis_row | None, file_row | None)."""
    a_row = require_analysis(db, analysis_id, key) if analysis_id else None
    if file_id is None and a_row is not None:
        file_id = a_row.file_id
    f_row = require_file(db, file_id, key) if file_id else None
    if a_row is None and f_row is None:
        raise HTTPException(status_code=422, detail="provide either analysis_id or file_id")
    return a_row, f_row


# ---------------------------------------------------------------------------------------
# demodulation
# ---------------------------------------------------------------------------------------
class DemodRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    file_id: str | None = None
    analysis_id: str | None = None
    modulation: str = Field(..., description="BPSK | QPSK | 8PSK | 16QAM | 64QAM | 2FSK | GFSK | AM | FM")
    symbol_rate_hz: float | None = Field(default=None, gt=0, le=1e9)
    rolloff: float = Field(default=0.35, ge=0.05, le=1.0)
    carrier_hint_hz: float | None = None
    differential: bool = False
    region: dict[str, Any] | None = None
    format_override: dict[str, Any] | None = None
    max_samples: int | None = Field(default=None, ge=256, le=12_000_000)
    persist: bool = True


@router.post("/demodulate", summary="Run one demodulator and return symbols, bits and quality")
def demodulate(req: DemodRequest, key: str = Depends(session_key),
               db: Session = Depends(get_db)) -> dict:
    a_row, f_row = _resolve(key, db, req.analysis_id, req.file_id)
    mod_name = (req.modulation or "").upper()
    rs = req.symbol_rate_hz
    if rs is None and a_row is not None:
        res = analysis_result(a_row)
        info = bits_from_analysis(res)
        rs = info.get("symbol_rate_hz")
        if req.analysis_id and req.file_id is None:
            pass
    if rs is None and mod_name not in ("AM", "FM"):
        raise HTTPException(status_code=422,
                            detail=f"a symbol rate is required to demodulate {mod_name}; supply "
                                   "symbol_rate_hz or analyse the file first")
    try:
        prepared = prepare_samples(f_row, {"region": req.region, "format_override": req.format_override,
                                           "max_samples": req.max_samples}) if f_row else None
    except HTTPException:
        raise
    if prepared is None:
        raise HTTPException(status_code=422, detail="no samples could be loaded for this request")
    x, fs = prepared["samples"], prepared["fs"]
    if x.size < 64:
        raise HTTPException(status_code=422,
                            detail=f"only {x.size} samples are available after region selection")
    dm = demod_mod.demodulate(x, fs, mod_name, rs, rolloff=req.rolloff,
                              carrier_hint_hz=req.carrier_hint_hz, differential=req.differential)
    quality = dm.get("quality") or {}
    bits = dm.get("bits")
    payload = {
        "ok": bool(dm.get("ok")), "message": dm.get("error"),
        "modulation": mod_name, "symbol_rate_used_hz": dm.get("symbol_rate_used_hz"),
        "symbol_rate_requested_hz": rs, "n_symbols": dm.get("n_symbols"),
        "n_bits": int(np.size(bits)) if bits is not None else None,
        "quality": jsonable(quality), "evm_percent": quality.get("evm_percent"),
        "ber_estimate": dm.get("ber_estimate"), "bit_quality": dm.get("bit_quality"),
        "carrier_offset_hz": dm.get("carrier_offset_hz"),
        "timing_phase_frac": dm.get("timing_phase_frac"),
        "stages": trim(dm.get("stages") or [], max_list=32),
        "constellation": jsonable(_demod_constellation(dm)),
        "bits_preview": "".join(str(int(b)) for b in np.asarray(bits, dtype=np.int8)[:256])
        if bits is not None else None,
        "llr_available": dm.get("llrs") is not None,
        "input": {"n_samples": int(x.size), "fs": fs, "fs_known": prepared["fs_known"],
                  "region": prepared["region"], "notes": prepared["notes"]},
        "method": "carrier recovery -> symbol-rate refinement -> matched filter -> timing drift "
                  "correction -> decision; each stage reports its own status",
    }
    if req.persist and a_row is not None:
        row = _persist(db, DemodulationResult, {
            "analysis_id": a_row.id, "modulation": mod_name, "symbol_rate_hz": rs,
            "rolloff": req.rolloff, "ok": bool(dm.get("ok")), "message": dm.get("error"),
            "n_symbols": _num(dm.get("n_symbols")), "n_bits": payload["n_bits"],
            "evm_percent": _num(quality.get("evm_percent")), "evm_db": _num(quality.get("evm_db")),
            "ber_estimate": _num(dm.get("ber_estimate")), "bit_quality": dm.get("bit_quality"),
            "esn0_db": _num((dm.get("ber_estimate") or {}).get("esn0_db"))
            if isinstance(dm.get("ber_estimate"), dict) else None,
            "carrier_offset_hz": _num(dm.get("carrier_offset_hz")),
            "timing_phase_frac": _num(dm.get("timing_phase_frac")),
            "stages": jsonable(trim(dm.get("stages") or [], max_list=32)),
            "symbols_preview": jsonable(_demod_constellation(dm).get("i", [])[:128]),
            "bits_preview": payload["bits_preview"],
        })
        payload["demodulation_id"] = row.id
    return jsonable(payload)


def _demod_constellation(dm: dict) -> dict:
    syms = dm.get("symbols")
    if syms is None:
        return {}
    s = np.asarray(syms)
    s = s[np.isfinite(s)]
    if s.size > 6000:
        idx = np.linspace(0, s.size - 1, 6000).astype(int)
        s = s[idx]
    return {"i": np.real(s).tolist(), "q": np.imag(s).tolist(), "n_points": int(s.size),
            "view": "symbol samples from the demodulator (carrier + timing corrected)"}


# ---------------------------------------------------------------------------------------
# FEC / interleaving
# ---------------------------------------------------------------------------------------
class FecRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    analysis_id: str | None = None
    file_id: str | None = None
    max_bits: int = Field(default=8000, ge=256, le=100000)
    null_trials: int = Field(default=6, ge=1, le=40)
    try_rs: bool = True
    persist: bool = True


@router.post("/fec/analyze", summary="Automatic FEC family / code hypothesis testing")
def fec_analyze(req: FecRequest, key: str = Depends(session_key), db: Session = Depends(get_db)) -> dict:
    a_row, f_row = _resolve(key, db, req.analysis_id, req.file_id)
    bits, soft, meta = _bits_for(db, a_row, f_row, key)
    if bits is None or bits.size < 64:
        return {"ok": False, "hypotheses": [], "best": None,
                "message": "no demodulated bit stream is available for FEC analysis",
                "hint": "run a demodulation first (the Analyze pipeline does it automatically)"}
    res = fec_mod.analyse_fec(bits, soft=soft, max_bits=req.max_bits,
                              null_trials=req.null_trials, try_rs=req.try_rs)
    if req.persist and a_row is not None:
        for rank, h in enumerate((res.get("hypotheses") or [])[:8], start=1):
            metrics = h.get("metrics") or {}
            _persist(db, FECHypothesis, {
                "analysis_id": a_row.id, "family": h.get("family"), "hypothesis": str(h.get("hypothesis")),
                "rank": rank, "confidence": h.get("confidence"),
                "support": metrics.get("support") or h.get("support"),
                "z_score": metrics.get("z_score") or metrics.get("z"),
                "null_level": metrics.get("null_distance_fraction"),
                "n_blocks": metrics.get("blocks_tested") or metrics.get("n_blocks"),
                "n_valid": metrics.get("blocks_clean"),
                "n_corrected": metrics.get("blocks_corrected"),
                "n_clean": metrics.get("blocks_clean"),
                "parameters": jsonable(trim(h.get("code") or metrics, max_list=40)),
                "evidence": jsonable(h.get("evidence")),
                "limitations": jsonable(h.get("limitations")),
            })
    return jsonable({"ok": res.get("ok"), "best": trim(res.get("best") or {}, max_list=40),
                     "hypotheses": trim(res.get("hypotheses") or [], max_list=24),
                     "notes": res.get("notes"), "n_bits_tested": int(bits.size),
                     "bit_source": meta,
                     "policy": "a hypothesis is only reported as 'best' when its confidence is at "
                               "least 0.20 and every claim lists its evidence and limitations"})


class InterleaveRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    analysis_id: str | None = None
    file_id: str | None = None
    max_bits: int = Field(default=3000, ge=512, le=40000)
    run_decode_test: bool = True
    persist: bool = True
    fec_hypothesis: str | None = None
    time_budget_s: float = Field(default=25.0, ge=2.0, le=600.0)


@router.post("/interleaving/analyze", summary="Automatic interleaver hypothesis testing")
def interleaving_analyze(req: InterleaveRequest, key: str = Depends(session_key),
                         db: Session = Depends(get_db)) -> dict:
    a_row, f_row = _resolve(key, db, req.analysis_id, req.file_id)
    bits, soft, meta = _bits_for(db, a_row, f_row, key)
    if bits is None or bits.size < 128:
        return {"ok": False, "hypotheses": [], "best": None,
                "message": "no demodulated bit stream is available for interleaver analysis"}
    fec_result = None
    if a_row is not None:
        res = analysis_result(a_row)
        fec_result = ((res.get("signal") or {}).get("fec")) or None
        if req.fec_hypothesis and fec_result is not None:
            match = [h for h in (fec_result.get("hypotheses") or [])
                     if str(h.get("hypothesis", "")).startswith(req.fec_hypothesis)]
            if match:
                fec_result = dict(fec_result, best=match[0])
    res = int_mod.analyse_interleaving(soft=soft, bits=bits, fec_result=fec_result,
                                       max_bits=req.max_bits, run_decode_test=req.run_decode_test,
                                       time_budget_s=req.time_budget_s)
    if req.persist and a_row is not None:
        for rank, h in enumerate((res.get("hypotheses") or [])[:8], start=1):
            _persist(db, InterleavingHypothesis, {
                "analysis_id": a_row.id, "kind": h.get("kind"), "hypothesis": str(h.get("hypothesis")),
                "rank": rank, "confidence": h.get("confidence"),
                "dispersion": h.get("dispersion"), "z_score": h.get("z_score"),
                "improvement_per_bit": h.get("improvement_per_bit"),
                "relative_improvement": h.get("relative_improvement"),
                "code": str(h.get("code") or h.get("hypothesis"))[:64],
                "parameters": jsonable(trim(h.get("rows") or h.get("geometry") or {}, max_list=20)),
                "evidence": jsonable(h.get("evidence")), "limitations": jsonable(h.get("limitations")),
            })
    return jsonable({"ok": res.get("ok"), "best": trim(res.get("best") or {}, max_list=40),
                     "hypotheses": trim(res.get("hypotheses") or [], max_list=24),
                     "notes": res.get("notes"), "cluster_test": trim(res.get("cluster_test") or {},
                                                                      max_list=20),
                     "bit_source": meta, "n_bits_tested": int(bits.size),
                     "policy": "interleaver geometry is only claimed when de-interleaving with that "
                               "geometry makes the FEC decoder measurably better than a random "
                               "permutation control group"})


def _bits_for(db: Session, a_row: AnalysisSession | None, f_row: UploadedFile | None, key: str):
    """Prefer the bit stream of the analysis; otherwise demodulate the file with its own estimate."""
    if a_row is not None:
        res = analysis_result(a_row)
        info = bits_from_analysis(res)
        if info["bits"] is not None and info["bits"].size >= 64:
            return info["bits"], info["soft"], {
                "source": "analysis", "analysis_id": a_row.id, "modulation": info["modulation"],
                "symbol_rate_hz": info["symbol_rate_hz"], "persisted": True}
    if f_row is None:
        return None, None, {}
    prepared = prepare_samples(f_row, {"max_samples": 2_000_000})
    res = analysis_result(a_row) if a_row is not None else None
    sig = (res or {}).get("signal") or {}
    mod = ((sig.get("modulation") or {}).get("primary")) or "QPSK"
    rs = sig.get("symbol_rate_hz")
    if not rs:
        return None, None, {"source": "file", "reason": "no symbol rate available to demodulate"}
    dm = demod_mod.demodulate(prepared["samples"], prepared["fs"], mod, rs)
    bits = dm.get("bits")
    if bits is None:
        return None, None, {"source": "file", "reason": dm.get("error") or "demodulation failed"}
    return (np.asarray(bits, dtype=np.int8), dm.get("llrs"),
            {"source": "file (re-demodulated)", "modulation": mod, "symbol_rate_hz": rs,
             "n_bits": int(np.size(bits))})


# ---------------------------------------------------------------------------------------
# correlation
# ---------------------------------------------------------------------------------------
class CorrelateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    analysis_id: str | None = None
    file_id: str | None = None
    pattern: str | None = Field(default=None, description="ASCII text | hex string | 0/1 bit string")
    kind: str = Field(default="auto", description="auto | text | hex | bits")
    max_errors: int = Field(default=0, ge=0, le=64)
    auto: bool = True
    max_hits: int = Field(default=32, ge=1, le=256)
    stream: str = Field(default="auto", description="auto | demod | decoded")
    persist: bool = True


@router.post("/correlate", summary="Search the bit stream for a header / byte / hex / ASCII pattern")
def correlate(req: CorrelateRequest, key: str = Depends(session_key),
              db: Session = Depends(get_db)) -> dict:
    a_row, f_row = _resolve(key, db, req.analysis_id, req.file_id)
    bits, soft, meta = _bits_for(db, a_row, f_row, key)
    if bits is None or bits.size < 32:
        return {"ok": False, "message": "no demodulated bit stream is available for correlation",
                "searches": []}
    # The pattern is usually a payload-level marker, so the search runs on the error-corrected
    # information bits when the FEC engine produced them, and on the raw decisions otherwise; in
    # "auto" both streams are searched and each hit says which stream it came from.
    decoded_streams: list[tuple[str, np.ndarray]] = []
    if req.analysis_id and req.stream in ("auto", "decoded"):
        info = bits_from_analysis(analysis_result(a_row)) if a_row is not None else {}
        if info.get("decoded") is not None:
            decoded_streams.append((f"decoded ({info.get('decoded_from')})", info["decoded"]))
    if req.stream == "decoded" and not decoded_streams:
        return {"ok": False, "searches": [],
                "message": "no error-corrected bit stream is available; run the FEC analysis first "
                           "or use stream='demod'"}
    bits_raw = bits
    if req.stream in ("auto", "decoded") and decoded_streams:
        bits = decoded_streams[0][1]
    patterns = []
    if req.pattern:
        try:
            parsed = corr_mod.parse_pattern(req.pattern, kind=req.kind)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"the pattern could not be parsed: {exc}") from exc
        if parsed.get("ok") is False:
            raise HTTPException(status_code=422, detail=parsed.get("message") or "invalid pattern")
        patterns.append(dict(parsed, max_errors=req.max_errors, max_hits=req.max_hits))
    res = corr_mod.analyse_correlation(bits, patterns=patterns, auto=req.auto)
    res["stream_searched"] = ("decoded information bits" if decoded_streams and req.stream != "demod"
                              else "raw demodulator decisions")
    if req.stream == "auto" and decoded_streams:
        # also search the other stream so a hit can never be attributed to the wrong bit sequence
        other = "raw demodulator decisions"
        res_other = corr_mod.analyse_correlation(bits_raw, patterns=patterns, auto=False)
        if any(s.get("n_hits") for s in (res_other.get("searches") or [])):
            res["note_other_stream"] = (f"the pattern was not found in the {other}, only in the "
                                        "error-corrected stream")
            res["other_stream_searches"] = trim(res_other.get("searches") or [], max_list=8)
    if req.persist and a_row is not None:
        for s in (res.get("searches") or []):
            _persist(db, CorrelationResult, {
                "analysis_id": a_row.id, "pattern_text": req.pattern, "pattern_kind": s.get("kind"),
                "pattern_bits": s.get("pattern_bits"), "max_errors": req.max_errors,
                "n_hits": s.get("n_hits"), "n_occurrences": s.get("n_occurrences"),
                "score": s.get("score"), "best_position_bits": s.get("best_position_bits"),
                "best_matches": s.get("best_matches"),
                "positions": jsonable(trim(s.get("hits") or [], max_list=64)),
                "auto_candidates": jsonable(trim((res.get("auto") or {}).get("candidates") or [],
                                                 max_list=32)),
                "notes": jsonable(s.get("notes")),
            })
    return jsonable({"ok": res.get("ok"), "searches": trim(res.get("searches") or [], max_list=32),
                     "auto": trim(res.get("auto") or {}, max_list=32),
                     "summary": res.get("summary"), "n_bits": int(bits.size),
                     "bit_source": meta,
                     "interpretation": "a hit means the pattern's bits were found at that position in "
                                       "the demodulated stream; it is evidence about structure, not "
                                       "a claim about the protocol"})


# ---------------------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------------------
class CompareRequest(BaseModel):
    analysis_a: str
    analysis_b: str


@router.post("/compare", summary="Compare two analyses (similar / partially / different)")
def compare(req: CompareRequest, key: str = Depends(session_key), db: Session = Depends(get_db)) -> dict:
    a_row = require_analysis(db, req.analysis_a, key)
    b_row = require_analysis(db, req.analysis_b, key)
    res_a = analysis_result(a_row)
    res_b = analysis_result(b_row)
    res_a["_analysis"] = {"id": a_row.id}
    res_b["_analysis"] = {"id": b_row.id}
    f_a = db.get(UploadedFile, a_row.file_id) if a_row.file_id else None
    f_b = db.get(UploadedFile, b_row.file_id) if b_row.file_id else None
    row, res = compare_results(db, key, res_a, res_b, f_a, f_b)
    return jsonable({"comparison_id": row.id, "verdict": row.verdict, "similarity": row.similarity,
                     "metrics": trim(res.get("metrics") or {}, max_list=200),
                     "notes": res.get("notes"), "a": {"analysis_id": a_row.id,
                                                      "file": f_a.filename if f_a else None},
                     "b": {"analysis_id": b_row.id, "file": f_b.filename if f_b else None},
                     "policy": "a verdict of 'similar' means the measured parameters agree within "
                               "the stated tolerances - it is never a claim about the source"})


@router.get("/comparisons", summary="Stored comparisons")
def comparisons(limit: int = Query(25, ge=1, le=200), key: str = Depends(session_key),
                db: Session = Depends(get_db)) -> dict:
    rows = db.scalars(select(Comparison).where(Comparison.session_key == key)
                      .order_by(Comparison.created_at.desc()).limit(limit)).all()
    return {"comparisons": [{"comparison_id": r.id, "verdict": r.verdict, "similarity": r.similarity,
                             "metrics": r.metrics, "notes": r.notes,
                             "analysis_a": r.analysis_a_id, "analysis_b": r.analysis_b_id,
                             "created_at": r.created_at.isoformat() if r.created_at else None}
                            for r in rows]}
