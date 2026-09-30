"""Single-stage DSP endpoints plus the API aliases used by the UI, and message-recovery
verification for generated signals.

Each endpoint runs exactly one stage of the chain on freshly loaded (or previously stored) samples,
so a user can inspect FFT, detection, classification, symbol rate, constellation or eye diagram
without paying for the whole pipeline - and the numbers always come from the same DSP modules the
pipeline uses.
"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from dsp import bitstream as bits_mod
from dsp import correlate as corr_mod
from dsp import demod as demod_mod
from dsp import detect as detect_mod
from dsp import modulation as mod_mod
from dsp import params as params_mod
from dsp import preprocess as pre_mod
from dsp import spectrum as spec_mod
from dsp import synth as synth_mod
from dsp import waterfall as wf_mod

from ..db import get_db
from ..models import AnalysisSession, GeneratedSignal, Report
from ..reports import build_report
from ..security import rate_limit
from ..views import trim
from .common import (analysis_result, bits_from_analysis, jsonable, prepare_samples,
                     require_analysis, require_file, session_key)

router = APIRouter(tags=["modules"])


class StageRequest(BaseModel):
    """Common input: a stored file (optionally restricted to a band/time region)."""

    model_config = ConfigDict(populate_by_name=True)

    file_id: str | None = None
    analysis_id: str | None = None
    region: dict | None = None
    format_override: dict | None = None
    max_samples: int | None = Field(default=None, ge=256, le=12_000_000)


def _load(key: str, db: Session, req: StageRequest) -> tuple[dict, dict]:
    a_row = require_analysis(db, req.analysis_id, key) if req.analysis_id else None
    file_id = req.file_id or (a_row.file_id if a_row else None)
    if not file_id:
        raise HTTPException(status_code=422, detail="provide file_id or analysis_id")
    f_row = require_file(db, file_id, key)
    region = req.region
    if region is None and a_row is not None:
        res = analysis_result(a_row)
        seg = res.get("segmentation") or {}
        if seg.get("f_lo_hz") is not None and res.get("segment_center_hz") is not None:
            # reuse the emission the analysis selected
            region = {"f_lo_hz": seg.get("f_lo_in_hz"), "f_hi_hz": seg.get("f_hi_in_hz")}
            region = {k: v for k, v in region.items() if v is not None}
    prepared = prepare_samples(f_row, {"region": region, "format_override": req.format_override,
                                       "max_samples": req.max_samples})
    if prepared["samples"].size < 64:
        raise HTTPException(status_code=422,
                            detail=f"only {prepared['samples'].size} samples are available after the "
                                   "region selection; at least 64 are required")
    return prepared, {"file_id": file_id, "analysis_id": req.analysis_id, "region": prepared["region"]}


# ---------------------------------------------------------------------------------------
# preprocessing / time domain
# ---------------------------------------------------------------------------------------
class PreprocessRequest(StageRequest):
    steps: dict | None = None
    max_points: int = Field(default=4000, ge=128, le=20000)


@router.post("/preprocess", summary="Run the preprocessing chain and return before/after views")
def preprocess(req: PreprocessRequest, key: str = Depends(session_key),
               db: Session = Depends(get_db)) -> dict:
    prepared, meta = _load(key, db, req)
    x, fs = prepared["samples"], prepared["fs"]
    prep = pre_mod.run_pipeline(x, fs, req.steps)
    y = np.asarray(prep.get("samples", x))
    from dsp import utils as utils_mod
    t = np.arange(min(x.size, y.size)) / fs
    out = {
        "ok": bool(prep.get("ok", True)), "input": meta,
        "steps": jsonable(prep.get("steps")), "log": jsonable(prep.get("log")),
        "metrics_before": jsonable(prep.get("metrics_before")),
        "metrics_after": jsonable(prep.get("metrics_after")),
        "flags": jsonable(prep.get("flags")), "fs_out": prep.get("fs_out"), "message": prep.get("message"),
        "waveform_before": {"t_s": utils_mod.to_jsonable(utils_mod.decimate_for_plot(t, req.max_points)),
                            "i": utils_mod.to_jsonable(utils_mod.decimate_for_plot(np.real(x), req.max_points)),
                            "q": utils_mod.to_jsonable(utils_mod.decimate_for_plot(np.imag(x), req.max_points))},
        "waveform_after": {"t_s": utils_mod.to_jsonable(utils_mod.decimate_for_plot(t, req.max_points)),
                           "i": utils_mod.to_jsonable(utils_mod.decimate_for_plot(np.real(y), req.max_points)),
                           "q": utils_mod.to_jsonable(utils_mod.decimate_for_plot(np.imag(y), req.max_points))},
        "note": "the chain applies only the enabled steps; every step records what it changed",
    }
    return jsonable(out)


# ---------------------------------------------------------------------------------------
# FFT / spectrum / spectrogram
# ---------------------------------------------------------------------------------------
class SpectrumRequest(StageRequest):
    nperseg: int | None = Field(default=None, ge=32, le=1 << 20)
    window: str = "hann"
    band_hz: list[float] | None = None
    max_points: int = Field(default=4000, ge=128, le=40000)


@router.post("/spectrum", summary="Welch PSD with noise floor, SNR, peak and occupied bandwidth")
def spectrum(req: SpectrumRequest, key: str = Depends(session_key), db: Session = Depends(get_db)) -> dict:
    prepared, meta = _load(key, db, req)
    x, fs = prepared["samples"], prepared["fs"]
    band = tuple(req.band_hz) if req.band_hz and len(req.band_hz) == 2 else None
    sp = spec_mod.analyse_spectrum(x, fs, nperseg=req.nperseg, window=req.window, band_hz=band)
    return jsonable({"ok": sp.get("ok"), "input": meta, "spectrum": trim(sp, max_list=req.max_points),
                     "arrays": trim(sp.get("arrays") or {}, max_list=req.max_points),
                     "nperseg": sp.get("nperseg"), "estimator": sp.get("estimator")})


class SpectrogramRequest(StageRequest):
    nperseg: int | None = Field(default=None, ge=32, le=8192)
    overlap: float = Field(default=0.75, ge=0.0, le=0.95)
    window: str = "hann"
    max_freq: int = Field(default=0, ge=0, le=2048)
    max_time: int = Field(default=0, ge=0, le=4096)


@router.post("/spectrogram", summary="STFT waterfall matrix (downsampled, rounded for transport)")
def spectrogram(req: SpectrogramRequest, key: str = Depends(session_key),
                db: Session = Depends(get_db)) -> dict:
    prepared, meta = _load(key, db, req)
    pay = wf_mod.waterfall_payload(prepared["samples"], prepared["fs"], nperseg=req.nperseg,
                                   overlap=req.overlap, window=req.window,
                                   max_w=req.max_time or 900, max_h=req.max_freq or 700)
    if not pay.get("ok"):
        return {"ok": False, "message": pay.get("message"), "input": meta}
    dbm = np.asarray(pay["db"], dtype=float)
    return jsonable({"ok": True, "input": meta, "db": np.round(dbm, 2).tolist(),
                     "freq_hz": np.round(pay["freq_hz"], 2).tolist(),
                     "times_s": np.round(pay["times_s"], 6).tolist(),
                     "shape": [int(dbm.shape[0]), int(dbm.shape[1])],
                     "orientation": "db[frequency_index][time_index]",
                     "meta": trim({k: v for k, v in pay.items()
                                   if k not in ("db", "freq_hz", "times_s")}, max_list=40)})


class DetectRequest(StageRequest):
    nperseg: int | None = None
    snr_threshold_db: float = Field(default=6.0, ge=0, le=60)
    max_signals: int = Field(default=32, ge=1, le=64)
    p_fa: float = Field(default=1e-3, gt=0, le=1.0)
    reference_center_hz: float = 0.0


@router.post("/detect-signals", summary="CFAR detection of every emission in the record")
def detect_signals(req: DetectRequest, key: str = Depends(session_key),
                   db: Session = Depends(get_db)) -> dict:
    prepared, meta = _load(key, db, req)
    det = detect_mod.detect_signals(prepared["samples"], prepared["fs"], nperseg=req.nperseg,
                                    snr_threshold_db=req.snr_threshold_db,
                                    max_signals=req.max_signals, p_fa=req.p_fa,
                                    reference_center_hz=req.reference_center_hz)
    return jsonable({"ok": det.get("ok"), "input": meta, "detection": trim(det, max_list=200),
                     "signals": trim(det.get("signals") or [], max_list=64),
                     "count": len(det.get("signals") or [])})


# ---------------------------------------------------------------------------------------
# parameters / classification / symbol rate
# ---------------------------------------------------------------------------------------
class ParamRequest(StageRequest):
    obw_hz: float | None = None
    snr_db: float | None = None
    spectral_flat: bool = False


@router.post("/extract-parameters", summary="Amplitude / phase / envelope / symbol-rate statistics")
def extract_parameters(req: ParamRequest, key: str = Depends(session_key),
                       db: Session = Depends(get_db)) -> dict:
    prepared, meta = _load(key, db, req)
    rep = params_mod.full_parameter_report(prepared["samples"], prepared["fs"], obw_hz=req.obw_hz,
                                           snr_db=req.snr_db, spectral_flat=req.spectral_flat)
    return jsonable({"ok": True, "input": meta, "report": trim(rep, max_list=200),
                     "parameters": trim(rep.get("prm") or rep.get("parameters") or [], max_list=64)})


class ClassifyRequest(StageRequest):
    rs: float | None = Field(default=None, gt=0, le=1e9)
    obw_hz: float | None = None
    snr_db: float | None = None
    rolloff: float = Field(default=0.35, ge=0.05, le=1.0)


@router.post("/modulation/classify", summary="Hybrid modulation classification with evidence")
def classify(req: ClassifyRequest, key: str = Depends(session_key),
             db: Session = Depends(get_db)) -> dict:
    prepared, meta = _load(key, db, req)
    x, fs = prepared["samples"], prepared["fs"]
    is_real = not np.iscomplexobj(x)
    amc = mod_mod.classify(x, fs, rs=req.rs, obw_hz=req.obw_hz, snr_db=req.snr_db,
                           rolloff=req.rolloff, is_real=is_real)
    return jsonable({"ok": amc.get("ok"), "input": meta, "modulation": trim(amc, max_list=200),
                     "primary": amc.get("primary"), "candidates": trim(amc.get("candidates") or [],
                                                                       max_list=32),
                     "ml": trim(amc.get("ml") or {}, max_list=20),
                     "note": "the primary candidate is a classification with a confidence; the "
                             "ranked alternatives and the evidence are always reported together"})


class SymbolRateRequest(StageRequest):
    obw_hz: float | None = None
    snr_db: float | None = None
    spectral_flat: bool = False


@router.post("/symbol-rate", summary="Blind symbol-rate estimation with cross-validated candidates")
def symbol_rate(req: SymbolRateRequest, key: str = Depends(session_key),
                db: Session = Depends(get_db)) -> dict:
    prepared, meta = _load(key, db, req)
    x, fs = prepared["samples"], prepared["fs"]
    res = params_mod.estimate_symbol_rate(x, fs, obw_hz=req.obw_hz, snr_db=req.snr_db,
                                          spectral_flat=req.spectral_flat)
    primary = (res.get("primary") or [None])
    if isinstance(primary, list):
        primary = primary[0] if primary else None
    extra = {}
    if primary and primary.get("symbol_rate_hz"):
        extra["timing_line"] = trim(params_mod.timing_line_quality(x, fs, primary["symbol_rate_hz"]),
                                    max_list=20)
    return jsonable({"ok": res.get("ok"), "input": meta, "symbol_rate": trim(res, max_list=64),
                     "primary": trim(primary or {}, max_list=32), **extra})


# ---------------------------------------------------------------------------------------
# constellation / eye
# ---------------------------------------------------------------------------------------
class ConstellationRequest(StageRequest):
    modulation: str = "QPSK"
    symbol_rate_hz: float | None = Field(default=None, gt=0)
    rolloff: float = Field(default=0.35, ge=0.05, le=1.0)
    view: str = Field(default="symbols", pattern="^(raw|filtered|symbols)$")
    max_points: int = Field(default=2500, ge=100, le=20000)


@router.post("/constellation", summary="I/Q scatter (raw / filtered / symbol-sampled) with EVM")
def constellation(req: ConstellationRequest, key: str = Depends(session_key),
                  db: Session = Depends(get_db)) -> dict:
    prepared, meta = _load(key, db, req)
    rs = req.symbol_rate_hz
    estimated = None
    if not rs:
        est = params_mod.estimate_symbol_rate(prepared["samples"], prepared["fs"])
        primary = est.get("primary") or []
        if isinstance(primary, list):
            primary = primary[0] if primary else {}
        rs = (primary or {}).get("symbol_rate_hz")
        estimated = trim(primary or {}, max_list=20)
    if not rs:
        return {"ok": False, "input": meta, "constellation": None,
                "message": "Constellation unavailable: a symbol rate is required and could not be "
                           "estimated reliably for this record.",
                "what_to_try": "supply symbol_rate_hz, or analyse a longer record of a digital signal"}
    cv = mod_mod.constellation_view(prepared["samples"], prepared["fs"], req.modulation.upper(),
                                    float(rs), rolloff=req.rolloff, view=req.view,
                                    max_points=req.max_points)
    return jsonable({"ok": cv.get("ok"), "input": meta, "constellation": trim(cv, max_list=20000),
                     "symbol_rate_used_hz": float(rs), "symbol_rate_estimate": estimated,
                     "symbol_rate_source": "supplied" if req.symbol_rate_hz else "measured",
                     "error": cv.get("error")})


class EyeRequest(StageRequest):
    symbol_rate_hz: float | None = Field(default=None, gt=0)
    rolloff: float = Field(default=0.35, ge=0.05, le=1.0)
    span_symbols: int = Field(default=2, ge=1, le=4)
    n_traces: int = Field(default=120, ge=8, le=400)


@router.post("/eye-diagram", summary="Eye diagram and timing-quality metrics")
def eye_diagram(req: EyeRequest, key: str = Depends(session_key), db: Session = Depends(get_db)) -> dict:
    prepared, meta = _load(key, db, req)
    rs = req.symbol_rate_hz
    estimated = None
    if not rs:
        est = params_mod.estimate_symbol_rate(prepared["samples"], prepared["fs"])
        primary = est.get("primary") or []
        if isinstance(primary, list):
            primary = primary[0] if primary else {}
        rs = (primary or {}).get("symbol_rate_hz")
        estimated = trim(primary or {}, max_list=20)
    if not rs:
        return {"ok": False, "input": meta, "eye": None,
                "message": "Eye diagram unavailable for this signal: a symbol rate is required and "
                           "could not be estimated reliably.",
                "what_to_try": "supply symbol_rate_hz, or analyse a longer record of a digital signal"}
    ey = mod_mod.eye_diagram(prepared["samples"], prepared["fs"], rs, rolloff=req.rolloff,
                             span_symbols=req.span_symbols, n_traces=req.n_traces)
    return jsonable({"ok": ey.get("ok"), "input": meta, "eye": trim(ey, max_list=600,
                                                                    drop_heavy=False),
                     "symbol_rate_used_hz": rs, "symbol_rate_estimate": estimated,
                     "error": ey.get("error")})


# ---------------------------------------------------------------------------------------
# bitstream / correlation (POST forms of the analysis endpoints)
# ---------------------------------------------------------------------------------------
class BitstreamRequest(StageRequest):
    modulation: str | None = None
    symbol_rate_hz: float | None = Field(default=None, gt=0)
    rolloff: float = Field(default=0.35, ge=0.05, le=1.0)
    max_bits: int = Field(default=262144, ge=64, le=4_000_000)
    stream: str = Field(default="auto", pattern="^(auto|demod|decoded)$")


@router.post("/bitstream", summary="Bit/byte statistics of a demodulated stream")
def bitstream(req: BitstreamRequest, key: str = Depends(session_key),
              db: Session = Depends(get_db)) -> dict:
    if req.analysis_id:
        a_row = require_analysis(db, req.analysis_id, key)
        res = analysis_result(a_row)
        info = bits_from_analysis(res)
        bits = info["decoded"] if (req.stream != "demod" and info.get("decoded") is not None) else info["bits"]
        if bits is None:
            return {"ok": False, "analysis_id": req.analysis_id, "stream": None,
                    "message": "this analysis holds no demodulated bit stream",
                    "hint": "the demodulator did not produce bits for this record - check the "
                            "modulation and symbol-rate hypotheses, or supply "
                            "modulation/symbol_rate_hz explicitly",
                    "n_bits": 0}
        src = {"source": info.get("source"), "modulation": info.get("modulation"),
               "symbol_rate_hz": info.get("symbol_rate_hz"),
               "stream": "decoded" if bits is info.get("decoded") else "demod"}
    else:
        prepared, meta = _load(key, db, req)
        mod = (req.modulation or "").upper()
        rs = req.symbol_rate_hz
        if not mod:
            amc = mod_mod.classify(prepared["samples"], prepared["fs"], rolloff=req.rolloff)
            mod = (amc.get("primary") or "QPSK")
        if not rs:
            est = params_mod.estimate_symbol_rate(prepared["samples"], prepared["fs"])
            primary = est.get("primary") or []
            if isinstance(primary, list):
                primary = primary[0] if primary else {}
            rs = (primary or {}).get("symbol_rate_hz")
        if not rs:
            return {"ok": False, "message": "no symbol rate is available for this record",
                    "hint": "supply symbol_rate_hz (or analyse the record first and use its "
                            "analysis_id, which carries the measured symbol rate)",
                    "n_bits": 0}
        dm = demod_mod.demodulate(prepared["samples"], prepared["fs"], mod, rs, rolloff=req.rolloff)
        bits = np.asarray(dm.get("bits"), dtype=np.int8) if dm.get("bits") is not None else None
        if bits is None:
            return {"ok": False, "message": dm.get("error") or "demodulation produced no bits",
                    "hint": "verify the modulation and symbol rate, or select a narrower band "
                            "around the emission and analyse a longer record",
                    "modulation_used": mod, "symbol_rate_used_hz": rs, "n_bits": 0}
        src = {"source": "file (re-demodulated)", "modulation": mod, "symbol_rate_hz": rs,
               "stream": "demod"}
    arr = np.asarray(bits, dtype=np.int8).ravel()[: req.max_bits]
    data = np.packbits(arr[: (arr.size // 8) * 8].astype(np.uint8)).astype(np.uint8)
    return jsonable({
        "ok": True, "stream": src, "n_bits": int(arr.size),
        "bits_preview": "".join(str(int(b)) for b in arr[:512]),
        "hex_dump": bits_mod.hex_dump(data, max_bytes=256),
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
        "policy": "a printable run is a candidate reading of the bits, not a decoded message",
    })


class CorrelationRequest(BitstreamRequest):
    pattern: str = Field(..., min_length=1)
    kind: str = Field(default="auto", pattern="^(auto|text|hex|bits)$")
    max_errors: int = Field(default=0, ge=0, le=64)
    auto: bool = True


@router.post("/correlation", summary="Search the stream for a header/byte/hex/ASCII pattern")
def correlation(req: CorrelationRequest, key: str = Depends(session_key),
                db: Session = Depends(get_db)) -> dict:
    bs = bitstream(BitstreamRequest(**{**req.model_dump(), "max_bits": min(req.max_bits, 1_000_000)}),
                   key=key, db=db)
    from dsp import utils as utils_mod
    parsed = corr_mod.parse_pattern(req.pattern, kind=req.kind)
    if parsed.get("ok") is False:
        raise HTTPException(status_code=422, detail=parsed.get("message") or "invalid pattern")
    bits = utils_mod.bits_from_string(bs["bits_preview"]) if hasattr(utils_mod, "bits_from_string") else None
    return jsonable({"ok": True, "pattern": trim(parsed, max_list=20),
                     "preview_used": len(bs["bits_preview"]),
                     "hint": "run POST /api/correlate or GET /api/analysis/{id}/bitstream with a "
                             "pattern for a full-stream search (this endpoint reports the pattern "
                             "encoding and the stream statistics)",
                     "autocorrelation": bs["autocorrelation"],
                     "frame_candidates": bs["frame_candidates"]})


# ---------------------------------------------------------------------------------------
# API aliases used by the UI specification
# ---------------------------------------------------------------------------------------
@router.get("/history", summary="Alias of /api/analyses (analysis history)")
def history(file_id: str | None = None, limit: int = Query(100, ge=1, le=500),
            key: str = Depends(session_key), db: Session = Depends(get_db)) -> dict:
    stmt = select(AnalysisSession).where(AnalysisSession.session_key == key)
    if file_id:
        stmt = stmt.where(AnalysisSession.file_id == file_id)
    rows = db.scalars(stmt.order_by(AnalysisSession.created_at.desc()).limit(limit)).all()
    return {"history": [{"analysis_id": r.id, "file_id": r.file_id, "kind": r.kind, "mode": r.mode,
                         "status": r.status, "created_at": r.created_at.isoformat() if r.created_at else None,
                         "duration_ms": r.duration_ms, "summary": r.summary, "error": r.error}
                        for r in rows], "count": len(rows)}


@router.get("/analyze/{analysis_id}/status", summary="Alias: status of an analysis id")
def analyze_status(analysis_id: str, key: str = Depends(session_key),
                   db: Session = Depends(get_db)) -> dict:
    row = require_analysis(db, analysis_id, key)
    return {"analysis_id": row.id, "status": row.status, "progress": row.progress,
            "stage_message": row.stage_message, "error": row.error, "summary": row.summary,
            "duration_ms": row.duration_ms,
            "finished_at": row.finished_at.isoformat() if row.finished_at else None}


@router.get("/analyze/{analysis_id}", summary="Alias: full analysis result")
def analyze_get(analysis_id: str, key: str = Depends(session_key),
                db: Session = Depends(get_db)) -> dict:
    from .analysis import get_analysis
    return get_analysis(analysis_id, full=False, key=key, db=db)


# ---------------------------------------------------------------------------------------
# generated-signal verification (message recovery)
# ---------------------------------------------------------------------------------------
class VerifyRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    generated_id: str | None = None
    file_id: str | None = None
    analysis_id: str | None = None


def printable_runs(raw: bytes, min_len: int = 4) -> list[tuple[str, int]]:
    """Maximal printable-ASCII runs (>= ``min_len``) with their byte offsets - a measurement, not a guess."""
    out: list[tuple[str, int]] = []
    start = None
    for i, byte in enumerate(raw + b"\x00"):
        if 32 <= byte < 127:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_len:
                out.append((raw[start:i].decode("ascii", "replace"), start))
            start = None
    out.sort(key=lambda t: -len(t[0]))
    return out


def _align_and_score(expected: np.ndarray, recovered: np.ndarray, max_shift: int = 512) -> dict:
    """Best circular-free alignment of the recovered stream against the transmitted bits."""
    n = min(expected.size, recovered.size)
    if n < 32 or recovered.size < 32:
        return {"ok": False, "message": "not enough bits to compare"}
    probe = expected[: min(64, expected.size)]
    best = {"shift": 0, "matches": -1.0, "n": 0}
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            seg = recovered[shift:shift + probe.size]
        else:
            seg = recovered[: max(0, probe.size + shift)]
            if seg.size < probe.size:
                continue
        if seg.size < probe.size:
            continue
        score = float(np.mean(seg == probe))
        if score > best["matches"]:
            best = {"shift": shift, "matches": score, "n": int(seg.size)}
    shift = int(best["shift"])
    if shift >= 0:
        a, b = expected, recovered[shift:]
    else:
        a, b = expected[-shift:], recovered
    n = int(min(a.size, b.size))
    if n < 32:
        return {"ok": False, "message": "the recovered stream is too short after alignment"}
    a, b = a[:n], b[:n]
    matches = int(np.sum(a == b))
    return {"ok": True, "shift_bits": shift, "compared_bits": n, "matched_bits": matches,
            "recovery_percent": 100.0 * matches / n,
            "bit_error_rate": 1.0 - matches / n,
            "alignment_score": best["matches"]}


@router.post("/generator/verify",
             summary="Compare what a generated signal transmitted with what the analysis recovered")
def verify_generated(req: VerifyRequest, key: str = Depends(session_key),
                     db: Session = Depends(get_db)) -> dict:
    """Message-recovery check: ground truth of the generator vs the analysed bit stream.

    The comparison is bit-exact against the transmitted payload bits (after applying the same
    descrambler once, because the generator scrambles the payload before modulation), so the
    recovery percentage is a measurement and never a display value.
    """
    gen = None
    if req.generated_id:
        gen = db.get(GeneratedSignal, req.generated_id)
        if gen is None or gen.session_key != key:
            raise HTTPException(status_code=404, detail="generated signal not found in this session")
    if gen is None:
        if not (req.analysis_id or req.file_id):
            raise HTTPException(status_code=422,
                                detail="provide generated_id, or analysis_id/file_id to look it up")
        stmt = select(GeneratedSignal).where(GeneratedSignal.session_key == key)
        if req.file_id:
            stmt = stmt.where(GeneratedSignal.file_id == req.file_id)
        rows = db.scalars(stmt.order_by(GeneratedSignal.created_at.desc())).all()
        if req.analysis_id:
            a_row = require_analysis(db, req.analysis_id, key)
            rows = [r for r in rows if r.file_id == a_row.file_id] or rows
        gen = rows[0] if rows else None
    if gen is None:
        return {"ok": False, "message": "this analysis is not based on a signal generated in this "
                                        "session, so there is no transmitted ground truth to compare "
                                        "with",
                "note": "generate a signal on the Generator page (message text is stored in the "
                        "ground truth) and analyse that file"}

    gt = gen.ground_truth or {}
    bitstring = gt.get("payload_bitstring")
    if not bitstring:
        return {"ok": False, "message": "the generator stored no payload bit string for this signal"}
    expected = np.fromiter((1 if c == "1" else 0 for c in bitstring), dtype=np.int8)

    a_row = require_analysis(db, req.analysis_id, key) if req.analysis_id else None
    if a_row is None:
        rows = db.scalars(select(AnalysisSession)
                          .where(AnalysisSession.session_key == key,
                                 AnalysisSession.file_id == gen.file_id,
                                 AnalysisSession.status == "done")
                          .order_by(AnalysisSession.created_at.desc())).all()
        a_row = rows[0] if rows else None
    if a_row is None:
        return {"ok": False, "message": "no completed analysis of this generated file exists yet",
                "next_step": "POST /api/analyze with this file_id, then call verify again"}

    res = analysis_result(a_row)
    info = bits_from_analysis(res)
    used = "decoded" if info.get("decoded") is not None else "demod"
    recovered = info["decoded"] if used == "decoded" else info["bits"]
    if recovered is None or recovered.size < 32:
        return {"ok": False, "message": "the analysis produced no usable bit stream",
                "analysis_status": a_row.status}

    # The generator scrambles the payload (1 + x^9 + x^11) *before* modulation, so the recovered
    # stream is the scrambled bit sequence unless a FEC decode produced information bits.  Both
    # references are therefore scored and the better one is reported, with the reference named -
    # never silently assumed.
    scramble_used = bool((gen.spec or {}).get("scramble", True))
    references: list[tuple[str, np.ndarray]] = []
    if scramble_used:
        try:
            references.append(("scrambled payload (what the modulator actually transmitted)",
                               synth_mod.multiplicative_scramble(expected)))
        except Exception:                                                # pragma: no cover
            pass
        references.append(("payload bits (descrambled reference)", expected))
    else:
        references.append(("payload bits (the generator does not scramble)", expected))
    best_ref, best_score = references[0][0], _align_and_score(references[0][1], recovered)
    for name, ref in references[1:]:
        sc = _align_and_score(ref, recovered)
        if sc.get("ok") and (not best_score.get("ok")
                             or float(sc.get("matched_bits", 0)) > float(best_score.get("matched_bits", 0))):
            best_ref, best_score = name, sc
    score = best_score
    out: dict = {
        "ok": bool(score.get("ok")), "generated_id": gen.id, "analysis_id": a_row.id,
        "file_id": gen.file_id,
        "spec": jsonable(gen.spec), "stream_used": used,
        "stream_source": (info.get("decoded_from") if used == "decoded"
                          else "raw demodulator decisions (no FEC hypothesis was claimed)"),
        "expected_bits": int(expected.size), "recovered_bits": int(recovered.size),
        "scramble": bool((gen.spec or {}).get("scramble", True)),
        "scrambler_note": ("the generator scrambles the payload (1+x^9+x^11) before modulation; the "
                           "text below is the recovered stream with the matching descrambler applied"
                           if bool((gen.spec or {}).get("scramble", True)) else
                           "the generator did not scramble the payload"),
        "score": score,
        "reference": best_ref,
        "ran": True,
    }
    if score.get("ok"):
        n_cmp = int(score.get("compared_bits") or 0)
        out["n_bits_compared"] = n_cmp
        out["match_ratio"] = (round(float(score["matched_bits"]) / max(n_cmp, 1), 4) if n_cmp else None)
        out["best_shift_bits"] = int(score.get("shift_bits") or 0)
        out["match"] = bool(n_cmp and out["match_ratio"] >= 0.95)
        out["recovery_percent"] = round(float(score["recovery_percent"]), 3)
        out["text_match_percent"] = 100.0 if out.get("text_match") else 0.0
        out["verdict"] = ("exact recovery: every compared payload bit matches"
                          if score["matched_bits"] == score["compared_bits"] else
                          f"{score['matched_bits']} of {score['compared_bits']} payload bits match")
        shift = int(score.get("shift_bits") or 0)
        if used == "decoded":
            aligned = recovered[shift:] if shift >= 0 else recovered
        else:
            aligned = recovered[shift:] if shift >= 0 else recovered
        if out["scramble"]:
            aligned = synth_mod.multiplicative_descramble(aligned)
        n_bytes = int(min(aligned.size, expected.size) // 8)
        if n_bytes:
            raw = np.packbits(aligned[: n_bytes * 8].astype(np.uint8)).tobytes()
            text = "".join(chr(b) if 32 <= b < 127 else "." for b in raw[:200])
            out["recovered_bytes_hex"] = raw[:120].hex()
            out["recovered_ascii"] = text
        want_text = gt.get("payload_text")
        if want_text:
            # The expected text is only ever used as a *search key* in what the bits actually
            # contain: it is never echoed back as if it had been recovered.
            want = str(want_text)
            runs = printable_runs(raw) if n_bytes else []
            out["printable_ascii_candidates"] = [{"text": t, "byte_offset": int(o)}
                                                 for t, o in runs[:5]]
            found = next((t for t, _ in runs if want in t), None)
            out["expected_text"] = want
            out["text_match"] = bool(found)
            out["text_match_note"] = (f"the transmitted message was found verbatim in the recovered "
                                      f"stream at byte offset "
                                      f"{next(o for t, o in runs if want in t)}"
                                      if found else
                                      "the transmitted message was not found in the recovered bytes; "
                                      "the printable runs listed above are what the bits really contain")
    out["policy"] = ("recovery is measured bit-by-bit against the generator's ground truth; a text "
                     "match is only claimed when the recovered bytes literally contain the "
                     "transmitted message")
    return jsonable(out)
