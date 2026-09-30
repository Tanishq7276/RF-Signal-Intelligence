"""Analysis services: orchestrate the DSP layer, persist the results, build reports.

The API routes stay thin; everything that touches the DSP modules or the database lives here.
Results are persisted twice on purpose:

* **queryable metadata** (segments, parameters, hypotheses, bitstream summary) goes into the
  relational schema so history, comparison and reporting can be answered with SQL, and
* the **full result** (including the downsampled spectrum/waterfall arrays) is written to a
  compressed cache file next to the analysis, so nothing needs to be recomputed to redraw charts.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time
import uuid
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from dsp import benchmark as bench_mod
from dsp import compare as compare_mod
from dsp import pipeline as pipe
from dsp import preprocess as pre_mod
from dsp import scenarios as scen
from dsp import synth as synth_mod
from dsp import utils as utils_mod

from .config import settings
from .jobs import Job
from .models import (AnalysisSession, BenchmarkResult, Bitstream, Comparison, CorrelationResult,
                     DemodulationResult, FECHypothesis, GeneratedSignal, InterleavingHypothesis,
                     ModulationResult, Report, SignalParameters, SignalSegment, UploadedFile)
from .storage import cache_read, cache_write, effective_fs, load_samples


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def _f(value: Any) -> float | None:
    try:
        if value is None:
            return None
        v = float(value)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _i(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def analysis_summary(result: dict) -> dict:
    """Compact, list-view-friendly summary of an analysis result."""
    sig = result.get("signal") or {}
    det = result.get("detection") or {}
    mod = (sig.get("modulation") or {})
    demod = (sig.get("demodulation") or {})
    hyp = (result.get("signal") or {}).get("hypotheses") or {}
    best = hyp.get("best") or {}
    fec = ((sig.get("fec") or {}).get("hypotheses") or [{}])[0]
    inter = ((sig.get("interleaving") or {}).get("hypotheses") or [{}])[0]
    return {
        "ok": bool(result.get("ok")),
        "message": result.get("message"),
        "n_samples": result.get("n_samples"),
        "duration_s": ((result.get("signal") or {}).get("duration_s")
                       or result.get("duration_s")),
        "fs": result.get("fs"), "fs_known": result.get("fs_known"),
        "n_emissions": len(det.get("signals") or []),
        "modulation": mod.get("primary"),
        "modulation_confidence": _f(mod.get("confidence")),
        "symbol_rate_hz": _f((result.get("signal") or {}).get("symbol_rate_hz")),
        "snr_db": _f(((result.get("signal") or {}).get("spectrum") or {}).get("snr_db")),
        "obw_hz": _f(((result.get("signal") or {}).get("spectrum") or {}).get("obw_99_hz")),
        "demodulation_ok": bool(demod.get("ok")),
        "evm_percent": _f((demod.get("quality") or {}).get("evm_percent")),
        "best_hypothesis": best.get("modulation"),
        "best_hypothesis_score": _f(best.get("score")),
        "fec": (fec or {}).get("hypothesis"),
        "fec_confidence": _f((fec or {}).get("confidence")),
        "interleaving": (inter or {}).get("hypothesis"),
        "confidence_overall": _f(((sig.get("confidence") or {}).get("overall"))),
        "notes": (result.get("notes") or [])[:6],
        "warnings": (result.get("warnings") or [])[:6],
    }


# ---------------------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------------------
def run_analysis(db: Session, session_key: str, file_row: UploadedFile, options: dict,
                 kind: str = "standard", job: Job | None = None) -> AnalysisSession:
    """Run a full analysis for one file and persist it.  Returns the AnalysisSession row."""
    row = AnalysisSession(session_key=session_key, file_id=file_row.id, kind=kind,
                          mode="BLIND" if options.get("blind") else "AUTO",
                          status="running", options=options)
    db.add(row)
    db.commit()
    db.refresh(row)
    t0 = time.time()

    def progress(frac: float, text: str | None = None) -> None:
        # progress is written to the job *and* to the analysis row, so a page reload can still show
        # how far a running analysis has come
        if job:
            job.set_progress(frac, text)
        try:
            row.progress = float(max(0.0, min(1.0, frac)))
            if text:
                row.stage_message = str(text)[:200]
            db.commit()
        except Exception:
            db.rollback()

    def cancelled() -> bool:
        return bool(job and job.cancelled())

    try:
        loaded = load_samples(file_row, override=options.get("format_override"))
        fs, fs_known, fs_source = effective_fs(file_row, loaded, options.get("format_override"))
        x = loaded["samples"]
        if job:
            job.set_progress(0.05, f"loaded {x.size:,} samples ({fs_source})")
        file_info = {
            "filename": file_row.filename, "size_bytes": file_row.size_bytes,
            "format": file_row.detected_format, "dtype": file_row.dtype,
            "iq_layout": file_row.iq_layout, "channels": file_row.channels,
            "sample_rate": (fs if fs_known else None), "sample_rate_source": fs_source,
            "center_frequency": file_row.center_frequency,
            "load_notes": loaded.get("load_notes") or [],
            "n_samples": int(x.size), "is_complex": bool(loaded.get("is_complex")),
        }
        result = pipe.analyse_record(x, fs, file_info=file_info, options=options,
                                     progress=progress, cancelled=cancelled, fs_known=fs_known)
        if result.get("ok") is False and result.get("cancelled"):
            row.status = "cancelled"
            row.finished_at = _dt.datetime.now(_dt.timezone.utc)
            row.summary = {"ok": False, "message": "analysis cancelled by the client"}
            db.commit()
            return row
        result["options_used"] = options
        result["kind"] = kind
        result["mode"] = row.mode
        if job:
            job.set_progress(0.97, "persisting results")
        cache_path = cache_write(row.id, options, result)
        _persist(db, row, file_row, result)
        row.status = "done" if result.get("ok") else "failed"
        row.error = None if result.get("ok") else str(result.get("message"))
        row.cache_path = cache_path
        row.summary = analysis_summary(result)
        row.notes = {"notes": result.get("notes"), "warnings": result.get("warnings")}
    except Exception as exc:                                              # pragma: no cover
        row.status = "failed"
        row.error = f"{type(exc).__name__}: {exc}"
        row.summary = {"ok": False, "message": row.error}
    finally:
        row.duration_ms = round((time.time() - t0) * 1000.0, 1)
        row.finished_at = _dt.datetime.now(_dt.timezone.utc)
        db.commit()
    return row


def _persist(db: Session, row: AnalysisSession, file_row: UploadedFile, result: dict) -> None:
    """Write the queryable parts of an analysis result into the relational schema."""
    det = result.get("detection") or {}
    sig = result.get("signal") or {}
    sig_regions = det.get("signals") or []
    chosen = result.get("selected_signal") or {}
    seg_rows: dict[int, SignalSegment] = {}
    for k, s in enumerate(sig_regions):
        seg = SignalSegment(
            analysis_id=row.id, file_id=file_row.id, segment_index=k, label=s.get("id"),
            kind=("tone" if (s.get("bandwidth_hz") or 1e9) < 1e3 else
                  ("burst" if (s.get("duty_cycle") or 1) < 0.8 else "modulated")),
            f_lo_hz=_f(s.get("f_lo_hz")), f_hi_hz=_f(s.get("f_hi_hz")),
            center_frequency_hz=_f(s.get("center_frequency_hz")),
            bandwidth_hz=_f(s.get("bandwidth_hz")), peak_frequency_hz=_f(s.get("peak_frequency_hz")),
            t0_s=_f(s.get("t0_s")), t1_s=_f(s.get("t1_s")), duration_s=_f(s.get("duration_s")),
            duty_cycle=_f(s.get("duty_cycle")), peak_power_db=_f(s.get("peak_power_db")),
            mean_power_db=_f(s.get("mean_power_db")), snr_db=_f(s.get("snr_db")),
            confidence=_f(s.get("confidence")), n_bursts=_i(s.get("n_bursts")),
            selected=bool(s.get("id") == chosen.get("id")),
            meta={"evidence": s.get("evidence"), "limitations": s.get("limitations"),
                  "snr_method": s.get("snr_method"), "bursts": s.get("bursts"),
                  "effective_bandwidth_hz": s.get("effective_bandwidth_hz")},
        )
        db.add(seg)
        seg_rows[k] = seg
    db.flush()

    # parameters ----------------------------------------------------------------
    params = sig.get("params") or {}
    for group, plist in params.items():
        for p in (plist or []):
            if not isinstance(p, dict):
                continue
            val = p.get("value")
            db.add(SignalParameters(
                analysis_id=row.id, group=group, name=str(p.get("name"))[:96],
                value_num=_f(val),
                value_text=(None if isinstance(val, (int, float)) or val is None else str(val)[:400]),
                unit=(str(p.get("unit"))[:24] if p.get("unit") else None),
                status=p.get("status"), confidence=_f(p.get("confidence")),
                method=(str(p.get("method"))[:800] if p.get("method") else None),
                evidence=utils_mod.to_jsonable(p.get("evidence")),
                limitations=utils_mod.to_jsonable(p.get("limitations")),
            ))
    # a couple of record-level parameters for convenience in lists
    for name, value, unit, status, conf, method in (
            ("Sample rate", result.get("fs") if result.get("fs_known") else None, "Hz",
             "ok" if result.get("fs_known") else "unable", 1.0 if result.get("fs_known") else None,
             "file metadata" if result.get("fs_known") else "unknown - not present in the capture"),
            ("Record duration", result.get("n_samples") / max(result.get("fs") or 1.0, 1e-9)
             if result.get("fs_known") else None, "s",
             "ok" if result.get("fs_known") else "unable", 1.0 if result.get("fs_known") else None,
             "samples / sample rate"),
            ("Detected emissions", len(sig_regions), None, "ok", 0.9,
             "block-averaged CFAR detector on the spectrogram")):
        db.add(SignalParameters(analysis_id=row.id, group="record", name=name, value_num=_f(value),
                                unit=unit, status=status, confidence=conf, method=method,
                                evidence=None, limitations=None))

    # modulation ----------------------------------------------------------------
    mod = sig.get("modulation") or {}
    for k, c in enumerate(mod.get("candidates") or []):
        db.add(ModulationResult(
            analysis_id=row.id, modulation=str(c.get("modulation")), rank=k + 1,
            probability=_f(c.get("probability")), confidence=_f(c.get("confidence")),
            probability_ml=_f(c.get("probability_ml")), probability_dsp=_f(c.get("probability_dsp")),
            evm_percent=_f(c.get("evm_percent")), method=(mod.get("method") or "")[:600],
            evidence=utils_mod.to_jsonable(mod.get("evidence")),
            rules_fired=utils_mod.to_jsonable(c.get("rules_fired") or mod.get("rules_fired")),
            features=utils_mod.to_jsonable(mod.get("features")),
            ml_model=mod.get("ml_model"), ml_model_accuracy=_f(mod.get("ml_model_accuracy")),
        ))

    # demodulation --------------------------------------------------------------
    demod = sig.get("demodulation") or {}
    if demod:
        db.add(DemodulationResult(
            analysis_id=row.id, modulation=str(demod.get("modulation") or mod.get("primary") or ""),
            symbol_rate_hz=_f(demod.get("symbol_rate_hz") or sig.get("symbol_rate_hz")),
            rolloff=_f(demod.get("rolloff")), ok=bool(demod.get("ok")),
            message=(str(demod.get("message"))[:600] if demod.get("message") else None),
            n_symbols=_i(demod.get("n_symbols")), n_bits=_i(demod.get("n_bits")),
            evm_percent=_f((demod.get("quality") or {}).get("evm_percent")),
            evm_db=_f((demod.get("quality") or {}).get("evm_db")),
            ber_estimate=_f((demod.get("ber_estimate") or {}).get("ber_estimate")),
            esn0_db=_f((demod.get("ber_estimate") or {}).get("esn0_db")),
            bit_quality=str(demod.get("bit_quality") or "")[:24],
            carrier_offset_hz=_f(demod.get("carrier_offset_hz")),
            timing_phase_frac=_f(demod.get("timing_phase_frac")),
            stages=utils_mod.to_jsonable(demod.get("stages")),
        ))

    # FEC / interleaving --------------------------------------------------------
    for k, h in enumerate((sig.get("fec") or {}).get("hypotheses") or []):
        db.add(FECHypothesis(
            analysis_id=row.id, family=h.get("family"), hypothesis=str(h.get("hypothesis"))[:96],
            rank=k + 1, confidence=_f(h.get("confidence")), support=_f(h.get("support")),
            z_score=_f(h.get("z_score") or h.get("z")), null_level=_f(h.get("null_level")),
            n_blocks=_i(h.get("n_blocks")), n_valid=_i(h.get("n_valid")),
            n_corrected=_i(h.get("n_corrected")), n_clean=_i(h.get("n_clean")),
            parameters=utils_mod.to_jsonable(h.get("parameters") or h.get("params")),
            evidence=utils_mod.to_jsonable(h.get("evidence")),
            limitations=utils_mod.to_jsonable(h.get("limitations")),
        ))
    for k, h in enumerate((sig.get("interleaving") or {}).get("hypotheses") or []):
        db.add(InterleavingHypothesis(
            analysis_id=row.id, kind=h.get("kind"), hypothesis=str(h.get("hypothesis"))[:96],
            rank=k + 1, confidence=_f(h.get("confidence")), dispersion=_f(h.get("dispersion")),
            z_score=_f(h.get("z_score")), improvement_per_bit=_f(h.get("improvement_per_bit")),
            relative_improvement=_f(h.get("relative_improvement")), code=h.get("code"),
            parameters=utils_mod.to_jsonable(h.get("parameters")),
            evidence=utils_mod.to_jsonable(h.get("evidence")),
            limitations=utils_mod.to_jsonable(h.get("limitations")),
        ))

    # bitstream ------------------------------------------------------------------
    bs = sig.get("bitstream") or {}
    if bs.get("n_bits"):
        db.add(Bitstream(
            analysis_id=row.id, n_bits=_i(bs.get("n_bits")),
            n_bits_analysed=_i(bs.get("n_bits_analysed") or bs.get("n_bits")),
            entropy_per_bit=_f(bs.get("bit_entropy_per_bit")),
            ones_fraction=_f(bs.get("ones_fraction")),
            bit_preview=(bs.get("bit_preview") or "")[:4096],
            hex_preview=(bs.get("hex_preview") or "")[:4096],
            byte_histogram=utils_mod.to_jsonable(bs.get("byte_histogram")),
            run_length=utils_mod.to_jsonable(bs.get("run_length")),
            periodic=utils_mod.to_jsonable(bs.get("periodicity") or bs.get("periodic")),
            frame_candidates=utils_mod.to_jsonable(bs.get("frame_boundaries")
                                                   or bs.get("frame_boundaries_candidates")),
            printable_ascii=utils_mod.to_jsonable(bs.get("printable_ascii")),
            autocorrelation=utils_mod.to_jsonable(bs.get("autocorrelation")),
            method=str(bs.get("method") or "demodulated hard decisions")[:400],
        ))

    # correlation -----------------------------------------------------------------
    corr = sig.get("correlation") or {}
    for s in corr.get("searches") or []:
        pat = s.get("pattern") or {}
        hits = s.get("hits") or []
        db.add(CorrelationResult(
            analysis_id=row.id, pattern_text=str(pat.get("text") or "")[:200],
            pattern_kind=pat.get("kind"), pattern_bits=_i(pat.get("n_bits")),
            max_errors=_i(pat.get("max_errors")), n_hits=len(hits),
            n_occurrences=len([h for h in hits if h.get("exact")]),
            score=_f(s.get("score")), best_position_bits=_i(hits[0].get("bit_offset") if hits else None),
            best_matches=_i(hits[0].get("matches") if hits else None),
            positions=utils_mod.to_jsonable(hits[:200]), auto_candidates=None,
            notes=utils_mod.to_jsonable(s.get("notes")),
        ))
    auto = (corr.get("auto") or {})
    if auto.get("candidates"):
        db.add(CorrelationResult(
            analysis_id=row.id, pattern_text="(automatic repeated-structure search)",
            pattern_kind="auto", n_hits=len(auto["candidates"]),
            score=_f(auto["candidates"][0].get("confidence")),
            positions=utils_mod.to_jsonable(auto["candidates"][:8]),
            auto_candidates=utils_mod.to_jsonable(auto.get("candidates")),
            notes=utils_mod.to_jsonable(auto.get("notes")),
        ))


# ---------------------------------------------------------------------------------------
# compare / benchmark / generation
# ---------------------------------------------------------------------------------------
def compare_results(db: Session, session_key: str, a: dict, b: dict,
                    file_a: UploadedFile | None, file_b: UploadedFile | None) -> Comparison:
    res = compare_mod.compare_results(a, b)
    row = Comparison(session_key=session_key, file_a_id=file_a.id if file_a else None,
                     file_b_id=file_b.id if file_b else None,
                     analysis_a_id=(a.get("_analysis") or {}).get("id"),
                     analysis_b_id=(b.get("_analysis") or {}).get("id"),
                     verdict=res.get("verdict"), similarity=_f(res.get("similarity")),
                     metrics=utils_mod.to_jsonable(res.get("metrics")),
                     notes=utils_mod.to_jsonable(res.get("notes")))
    db.add(row)
    db.commit()
    db.refresh(row)
    res["comparison_id"] = row.id
    return row, res


def run_benchmark(db: Session, session_key: str, config: dict, job: Job | None = None) -> BenchmarkResult:
    def progress(frac: float, text: str) -> None:
        if job:
            job.set_progress(frac, text)

    def cancelled() -> bool:
        return bool(job and job.cancelled())

    t0 = time.time()
    res = bench_mod.run_benchmark(config=config, progress=progress, cancelled=cancelled)
    row = BenchmarkResult(
        session_key=session_key, kind=config.get("kind", "full"),
        status="cancelled" if res.get("cancelled") else "done", config=config,
        accuracy=_f(res.get("accuracy")), confusion_matrix=utils_mod.to_jsonable(res.get("confusion_matrix")),
        classes=utils_mod.to_jsonable(res.get("classes")), per_class=utils_mod.to_jsonable(res.get("per_class")),
        metrics=utils_mod.to_jsonable({k: v for k, v in res.items()
                                       if k not in ("confusion_matrix", "classes", "per_class",
                                                    "config")}),
        n_cases=_i(res.get("n_cases")), duration_ms=round((time.time() - t0) * 1000.0, 1),
        notes=utils_mod.to_jsonable(res.get("notes")))
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def generate_signal(db: Session, session_key: str, spec: dict, fmt: str = "iq",
                    subdir: str | None = None) -> tuple[UploadedFile, GeneratedSignal, dict]:
    """Generate a synthetic signal, write it to disk and register it as a file."""
    gen = synth_mod.generate_signal(spec)
    file_id = uuid.uuid4().hex
    ext = ".wav" if fmt.lower() in ("wav", "wave") else ".iq"
    name = utils_mod_safe_name(spec, ext)
    dest = os.path.join(settings.generated_dir, session_key)
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, f"{file_id}__{name}")
    if ext == ".wav":
        synth_mod.write_wav_iq(path, gen.samples, gen.fs, spec.get("wav_bits", 16))
    else:
        synth_mod.write_iq(path, gen.samples, spec.get("dtype", "int16"))
    from dsp import iqformats as iqf
    spec_det = iqf.detect_format(path, name)
    spec_det["file_id"] = file_id
    spec_det["stored_path"] = path
    spec_det["source"] = "generated"
    from .storage import file_model_from_spec
    file_row = file_model_from_spec(session_key, spec_det)
    db.add(file_row)
    # flush the file row first: generated_signals.file_id is a foreign key to uploaded_files and
    # SQLAlchemy does not know the insert order without a relationship, so an un-flushed insert
    # crashes on SQLite with FOREIGN KEY constraint failed
    db.flush()
    gen_row = GeneratedSignal(session_key=session_key, file_id=file_id, spec=spec,
                              ground_truth=utils_mod.to_jsonable(gen.ground_truth),
                              seed=spec.get("seed"), n_samples=int(gen.samples.size),
                              duration_s=float(gen.samples.size / gen.fs))
    db.add(gen_row)
    db.commit()
    db.refresh(file_row)
    db.refresh(gen_row)
    return file_row, gen_row, spec_det


def utils_mod_safe_name(spec: dict, ext: str) -> str:
    mod = str(spec.get("modulation", "signal")).lower().replace(" ", "")
    rs = spec.get("symbol_rate")
    snr = spec.get("snr_db")
    parts = [mod]
    if rs:
        parts.append(f"{float(rs) / 1000:.0f}k" if float(rs) >= 1000 else f"{int(rs)}")
    if snr is not None:
        parts.append(f"{int(float(snr))}db")
    if spec.get("fec") and spec["fec"] != "none":
        parts.append(str(spec["fec"]).replace("conv_", "").replace("_", ""))
    if spec.get("interleaver") and spec["interleaver"] != "none":
        parts.append(str(spec["interleaver"]))
    return "_".join(parts)[:80] + ext


def load_demo_signal(db: Session, session_key: str, name: str) -> tuple[UploadedFile, dict] | None:
    """Register one of the shipped demo signals (or generate it if the file is missing)."""
    from dsp import iqformats as iqf
    from .storage import file_model_from_spec, save_stream
    path = os.path.join(settings.sample_dir, name)
    existing = db.scalar(select(UploadedFile).where(UploadedFile.session_key == session_key,
                                                    UploadedFile.filename == name,
                                                    UploadedFile.source == "demo"))
    if existing:
        return existing, {"reused": True}
    if os.path.exists(path):
        safe = name
        with open(path, "rb") as fh:
            spec = save_stream(session_key, safe, fh, source="demo")
        row = file_model_from_spec(session_key, spec)
        row.source = "demo"
        db.add(row)
        db.commit()
        db.refresh(row)
        return row, {"source": "sample_data"}
    # fall back to generating the demo signal on the fly so the button always works, even on a
    # checkout where sample_data/ was not shipped - the generated file is written back into
    # sample_data/ so the next call takes the normal path
    spec = scen.demo_spec(name)
    if spec is None:
        return None
    gen = synth_mod.generate_signal(spec)
    os.makedirs(settings.sample_dir, exist_ok=True)
    synth_mod.write_wav_iq(path, gen.samples, gen.fs, 16)
    with open(path, "rb") as fh:
        spec_det = save_stream(session_key, name, fh, source="demo")
    row = file_model_from_spec(session_key, spec_det)
    row.source = "demo"
    db.add(row)
    db.commit()
    db.refresh(row)
    return row, {"source": "generated from the shipped demo preset", "regenerated": True,
                 "file": name}
