"""End-to-end analysis pipeline.

``analyse_record`` is the single entry point used by the API, the demo flow and the benchmark:

    record -> preprocessing -> spectrum -> spectrogram -> detection
           -> per-emission segment -> parameters -> modulation classification
           -> demodulation -> FEC hypotheses -> interleaving hypotheses
           -> bitstream analysis -> correlation search -> hypotheses -> evidence graph

Every stage is logged with its status and duration so the UI can show progress and so a failure
anywhere is reported as a failed stage instead of an exception.  Nothing is fatal: an unsupported or
corrupt input produces a well-formed result with ``ok=False`` and an explanation.
"""

from __future__ import annotations

import time
import traceback

import numpy as np

from . import bitstream as bits_mod
from . import correlate as corr_mod
from . import demod as demod_mod
from . import detect as detect_mod
from . import fec as fec_mod
from . import interleave as inter_mod
from . import modulation as mod_mod
from . import params as params_mod
from . import preprocess as pre_mod
from . import spectrum as spec_mod
from . import waterfall as wf
from .hypothesis import analyse_hypotheses, hypothesis_param
from .utils import (clamp01, compact, db, human_si, param, safe_float, to_jsonable,
                    unavailable)

__all__ = ["analyse_record", "analyse_signal", "analyse_region", "demodulate_request",
           "fec_request", "interleaving_request", "correlate_request", "StageLog",
           "DEFAULT_OPTIONS"]

DEFAULT_OPTIONS: dict = {
    "preprocess": True,
    "preprocess_steps": None,          # None -> dsp.preprocess.DEFAULT_STEPS
    "detect": True,
    "spectrogram": True,
    "signal_id": None,                 # None -> the strongest emission
    "region": None,                    # {"f_lo_hz","f_hi_hz","t0_s","t1_s"}
    "modulation_hint": None,
    "symbol_rate_hint": None,
    "rolloff": 0.35,
    "run_fec": True,
    "run_interleaving": True,
    "run_correlation": True,
    "run_hypotheses": True,
    "run_eye": True,
    "max_bits": 200000,
    "max_fec_bits": 8000,
    "max_interleave_bits": 3000,        # bits used for the de-interleave-and-decode sweep
    "interleave_time_budget_s": 15.0,   # wall-clock budget for the interleaver catalogue search
    "patterns": [],
    "blind": False,
    "nperseg": None,
    "waterfall_max_w": 900,
    "waterfall_max_h": 700,
    "max_detect_signals": 16,
}


class StageLog:
    """Records each processing stage with its status, duration and a short note."""

    def __init__(self) -> None:
        self.stages: list[dict] = []
        self._t0 = time.time()

    def run(self, name: str, fn, *args, **kwargs):
        t0 = time.time()
        try:
            out = fn(*args, **kwargs)
            status = "ok" if (not isinstance(out, dict) or out.get("ok", True)) else "failed"
            note = ""
            if isinstance(out, dict):
                note = str(out.get("message") or out.get("error") or "")[:300]
            self.stages.append({"name": name, "status": status,
                                "duration_ms": round((time.time() - t0) * 1000.0, 1),
                                "note": note})
            return out
        except Exception as exc:                                        # pragma: no cover
            self.stages.append({"name": name, "status": "error",
                                "duration_ms": round((time.time() - t0) * 1000.0, 1),
                                "note": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc(limit=3)})
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def add(self, name: str, status: str, note: str = "") -> None:
        self.stages.append({"name": name, "status": status,
                            "duration_ms": round((time.time() - self._t0) * 1000.0, 1),
                            "note": note})

    def summary(self) -> dict:
        ok = sum(1 for s in self.stages if s["status"] == "ok")
        return {"n_stages": len(self.stages), "n_ok": ok,
                "n_failed": sum(1 for s in self.stages if s["status"] == "failed"),
                "n_error": sum(1 for s in self.stages if s["status"] == "error"),
                "total_ms": round(sum(s["duration_ms"] for s in self.stages), 1),
                "stages": self.stages}


# ---------------------------------------------------------------------------------------
# per-emission analysis
# ---------------------------------------------------------------------------------------
def _spectrum_arrays(spec: dict, max_points: int = 1800) -> dict | None:
    """Downsampled PSD arrays for the UI (mean for frequency, max-hold for level)."""
    f = np.asarray(spec.get("freq_hz") or [], dtype=np.float64)
    p = np.asarray(spec.get("psd_db") or [], dtype=np.float64)
    if f.size < 2 or f.size != p.size:
        return None
    decimated = False
    if f.size > max_points:
        step = int(np.ceil(f.size / float(max_points)))
        n = (f.size // step) * step
        f = f[:n].reshape(-1, step).mean(axis=1)
        p = p[:n].reshape(-1, step).max(axis=1)
        decimated = True
    return {"freq_hz": [float(v) for v in f], "psd_db": [float(v) for v in p],
            "decimated": decimated, "n_points": int(f.size),
            "unit": "dB (per FFT bin)"}


def analyse_signal(samples: np.ndarray, fs: float, region: dict | None = None,
                   options: dict | None = None, stage: StageLog | None = None,
                   progress=None, cancelled=None, fs_known: bool = True,
                   band_extracted: bool = False, noise_density_hint: float | None = None,
                   noise_source: str | None = None) -> dict:
    """Full analysis of one emission (already time/band sliced or the whole record)."""
    opt = dict(DEFAULT_OPTIONS)
    opt.update(options or {})
    st = stage or StageLog()
    x = np.asarray(samples)
    out: dict = {"ok": False, "fs": float(fs), "fs_known": bool(fs_known),
                 "n_samples": int(x.size),
                 "duration_s": float(x.size / fs) if fs else None,
                 "region": region, "notes": [], "warnings": []}
    if x.size < 128:
        out.update({"message": f"the selected segment holds only {x.size} samples; at least 128 "
                               "are needed for parameter estimation"})
        st.add("signal analysis", "failed", out["message"])
        return out
    if np.allclose(np.abs(x), 0.0):
        out.update({"message": "the selected segment is all zeros"})
        st.add("signal analysis", "failed", out["message"])
        return out
    is_real = not np.iscomplexobj(x)

    # ---- spectrum (band-restricted when a region is known) ---------------------------
    band = None
    if region and region.get("f_lo_hz") is not None and region.get("f_hi_hz") is not None:
        lo_r, hi_r = float(region["f_lo_hz"]), float(region["f_hi_hz"])
        pad_r = 0.6 * max(hi_r - lo_r, 1.0)
        band = (lo_r - pad_r, hi_r + pad_r)
    spec = st.run("spectrum analysis", spec_mod.analyse_spectrum, x, fs, opt["nperseg"],
                  band_hz=band, noise_density_hint=noise_density_hint,
                  noise_source=noise_source)
    snr_db = spec.get("snr_db") if spec.get("ok") else None
    obw = spec.get("obw_99_hz") if spec.get("ok") else None
    carrier = spec.get("center_frequency_hz") if spec.get("ok") else None
    out["spectrum"] = {
        "ok": bool(spec.get("ok")), "no_signal": bool(spec.get("no_signal", False)),
        "noise_floor_db": spec.get("noise_floor_db"), "noise_density": spec.get("noise_density"),
        "noise_reliable": spec.get("noise_reliable"), "numerical_floor": spec.get("numerical_floor"),
        "noise_source": spec.get("noise_source"),
        "peak_frequency_hz": spec.get("peak_frequency_hz"),
        "center_frequency_hz": spec.get("center_frequency_hz"),
        "frequency_offset_hz": spec.get("frequency_offset_hz"),
        "obw_99_hz": obw, "obw_lo_hz": spec.get("obw_lo_hz"), "obw_hi_hz": spec.get("obw_hi_hz"),
        "bw_3db_hz": spec.get("bw_3db_hz"), "bw_20db_hz": spec.get("bw_20db_hz"),
        "snr_db": snr_db, "snr_region_db": spec.get("snr_region_db"),
        "peak_db": spec.get("peak_db"), "peak_to_floor_db": spec.get("peak_to_floor_db"),
        "threshold_db": spec.get("threshold_db"), "line_like": spec.get("line_like"),
        "flat_top": spec.get("flat_top"), "occupied_fraction": spec.get("occupied_fraction"),
        "df_hz": spec.get("df_hz"), "nperseg": spec.get("nperseg"),
        "estimator": spec.get("estimator"), "dynamic_range_db": spec.get("dynamic_range_db"),
        "params": spec.get("params"), "evidence": spec.get("evidence", []),
        "limitations": spec.get("limitations", []),
        "message": spec.get("message"),
    }
    spec_arrays = _spectrum_arrays(spec)
    if spec_arrays:
        out["spectrum"]["arrays"] = spec_arrays
    if not spec.get("ok"):
        out["notes"].append(spec.get("message", "spectrum analysis failed"))

    # ---- parameters -----------------------------------------------------------------
    pr = st.run("parameter estimation", params_mod.full_parameter_report, x, fs,
                obw_hz=obw, snr_db=snr_db,
                spectral_flat=bool(spec.get("flat_top")))
    out["parameters"] = pr
    rs_est = (pr.get("symbol_rate") or {}).get("primary") if isinstance(pr, dict) else None
    rs = None
    if opt.get("symbol_rate_hint"):
        rs = float(opt["symbol_rate_hint"])
    elif isinstance(rs_est, dict):
        rs = rs_est.get("symbol_rate_hz")
    if rs is None and isinstance(pr.get("symbol_rate"), dict):
        alt = pr["symbol_rate"].get("alternatives") or []
        for a in alt:
            if a.get("symbol_rate_hz"):
                rs = a["symbol_rate_hz"]
                break
    out["symbol_rate_hz"] = rs
    mod_hint = opt.get("modulation_hint")

    # ---- modulation classification ---------------------------------------------------
    amc = st.run("modulation classification", mod_mod.classify, x, fs, rs=rs, obw_hz=obw,
                 snr_db=snr_db, rolloff=float(opt["rolloff"]),
                 is_real=is_real, carrier_hz=carrier)
    out["modulation"] = _trim_amc(amc)
    modulation = mod_hint or (amc.get("primary") if isinstance(amc, dict) else None)

    # ---- demodulation ----------------------------------------------------------------
    dm = None
    if modulation:
        dm = st.run(f"demodulation ({modulation})", demod_mod.demodulate, x, fs, modulation, rs,
                    rolloff=float(opt["rolloff"]),
                    carrier_hint_hz=carrier)
        if dm and dm.get("ok"):
            out["demodulation"] = {k: v for k, v in dm.items()
                                   if k not in ("symbols", "bits", "llrs", "stages")}
            out["demodulation"]["stages"] = compact(dm.get("stages", []))
            # The exact bit/LLR/symbol stream is kept in the cached result (under a private key that
            # the API layer strips) so that the FEC, interleaving, correlation, bitstream and
            # demodulation-reuse endpoints all work on the *same* stream the pipeline analysed
            # instead of re-deriving a slightly different one.
            syms = dm.get("symbols")
            out["_streams"] = {
                "modulation": modulation,
                "symbol_rate_hz": dm.get("symbol_rate_used_hz") or rs,
                "bits": [int(b) for b in np.asarray(dm.get("bits", []), dtype=np.int8).ravel()]
                if dm.get("bits") is not None else None,
                "llrs": ([float(v) for v in np.asarray(dm["llrs"], dtype=float).ravel()]
                         if dm.get("llrs") is not None else None),
                "symbols_i": np.real(np.asarray(syms)).tolist() if syms is not None else None,
                "symbols_q": np.imag(np.asarray(syms)).tolist() if syms is not None else None,
                "n_bits": int(np.size(dm.get("bits"))) if dm.get("bits") is not None else 0,
                "note": "raw demodulator output kept for the module endpoints; removed from the "
                        "browser payload (request /api/analysis/{id}/bitstream or /constellation)",
            }
            # the demodulator measures the symbol rate again (timing-drift estimation); when it
            # disagrees with the blind estimator its value is the better one and is reported
            rs_ref = dm.get("symbol_rate_used_hz")
            if rs_ref and rs and abs(rs_ref / rs - 1.0) > 1e-5:
                out["symbol_rate_hz"] = float(rs_ref)
                out["symbol_rate_note"] = (
                    f"the blind estimator reported {rs:,.1f} sym/s; the demodulator's timing-drift "
                    f"measurement refined it to {rs_ref:,.1f} sym/s "
                    f"({(rs_ref / rs - 1.0) * 1e6:+.0f} ppm), and the demodulated constellation "
                    f"was measured at that rate")
                rs = float(rs_ref)
        else:
            out["demodulation"] = {"ok": False, "message": (dm or {}).get("message")
                                   or (dm or {}).get("error", "demodulation failed")}
            out["warnings"].append(f"demodulation of {modulation} did not complete: "
                                   f"{out['demodulation'].get('message')}")
    else:
        out["demodulation"] = {"ok": False,
                               "message": "no modulation candidate was available to demodulate"}

    # ---- constellation / eye ---------------------------------------------------------
    if modulation and rs and modulation.upper() in mod_mod.LINEAR:
        # prefer the demodulator's own symbol samples: they are the symbols the bits were decided
        # from (carrier + timing corrected by the same chain that produced the bit stream), so the
        # constellation and the reported EVM describe one consistent signal path.  The blind view is
        # used only when demodulation did not succeed.
        from . import synth as _synth_mod
        cv = None
        if dm is not None and dm.get("ok") and dm.get("symbols") is not None:
            syms = np.asarray(dm["symbols"])
            syms = syms[np.isfinite(syms)]
            if syms.size >= 8:
                pts = syms[:2500]
                const = _synth_mod.constellation(modulation)
                cv = {"ok": True, "view": "symbols (demodulator output)",
                      "i": np.real(pts).tolist(), "q": np.imag(pts).tolist(),
                      "n_points": int(pts.size),
                      "reference_i": np.real(const).tolist(), "reference_q": np.imag(const).tolist(),
                      "quality": demod_mod.constellation_quality(pts, modulation),
                      "note": ("symbol samples taken from the demodulator (carrier recovered, timing "
                               "drift corrected, matched filter applied) - these are the symbols the "
                               "bit stream was decided from, so the EVM here and the demodulator EVM "
                               "describe the same signal path"),
                      "stages": [{"stage": "symbol source", "status": "ok",
                                  "detail": "demodulator symbol output (not a blind re-estimate)"}],
                      "downsampled": bool(syms.size > 2500)}
        if cv is None:
            cv = st.run("constellation analysis", mod_mod.constellation_view, x, fs, modulation, rs,
                        rolloff=float(opt["rolloff"]), view="symbols", max_points=2500)
        out["constellation"] = _trim_constellation(cv)
        if opt.get("run_eye"):
            out["eye"] = st.run("eye diagram", mod_mod.eye_diagram, x, fs, rs,
                                rolloff=float(opt["rolloff"]))
    else:
        out["constellation"] = {"ok": False,
                                "message": "constellation analysis requires a linear modulation "
                                           "and a symbol rate"}
        if modulation:
            out["eye"] = {"ok": False, "message": "eye diagrams are defined for linear modulations"}

    # ---- bitstream, FEC, interleaving, correlation -----------------------------------
    bits = None
    llrs = None
    if dm and dm.get("ok"):
        bits = dm.get("bits")
        llrs = dm.get("llrs")
    if bits is not None and np.size(bits) > 0:
        b = np.asarray(bits, dtype=np.uint8).ravel()[: int(opt["max_bits"])]
        out["bitstream"] = st.run("bitstream analysis", bits_mod.summarise_bits, b)
        if out["bitstream"].get("ok", True):
            out["bitstream"]["n_bits_analysed"] = int(b.size)
            out["bitstream"]["n_bits_total"] = int(np.size(bits))
        fec_res = None
        if opt.get("run_fec"):
            fec_res = st.run("FEC hypothesis testing", fec_mod.analyse_fec,
                             np.asarray(bits, dtype=np.int8), soft=llrs,
                             max_bits=int(opt["max_fec_bits"]), null_trials=4,
                             cancelled=cancelled)
            out["fec"] = _trim_fec(fec_res)
        if opt.get("run_interleaving"):
            soft = np.asarray(llrs, dtype=np.float64) if llrs is not None else None
            out["interleaving"] = _trim_inter(
                st.run("interleaving hypothesis testing", inter_mod.analyse_interleaving,
                       soft=soft, bits=np.asarray(bits, dtype=np.int8), fec_result=fec_res,
                       max_bits=min(int(opt.get("max_interleave_bits", 3000)), 8000),
                       time_budget_s=float(opt.get("interleave_time_budget_s", 20.0)),
                       cancelled=cancelled))
        if opt.get("run_correlation"):
            patterns = []
            for p in (opt.get("patterns") or []):
                parsed = corr_mod.parse_pattern(p.get("value"), kind=p.get("kind", "auto"))
                if parsed.get("ok"):
                    parsed["max_errors"] = int(p.get("max_errors", 0) or 0)
                    patterns.append(parsed)
            out["correlation"] = _trim_corr(
                st.run("correlation search", corr_mod.analyse_correlation,
                       np.asarray(bits, dtype=np.uint8), patterns=patterns,
                       max_errors=int(opt.get("max_errors", 0) or 0), auto=True))
    else:
        out["bitstream"] = {"ok": False,
                            "message": "no bits: demodulation did not produce a symbol stream"}
        out["fec"] = {"ok": False, "hypotheses": [], "message": "no bits available"}
        out["interleaving"] = {"ok": False, "hypotheses": [], "message": "no bits available"}
        out["correlation"] = {"ok": False, "searches": [], "message": "no bits available"}

    # ---- multi-hypothesis ------------------------------------------------------------
    if opt.get("run_hypotheses"):
        if progress:
            progress(0.6, "ranking hypotheses")
        out["hypotheses"] = st.run("multi-hypothesis analysis", analyse_hypotheses, x, fs,
                                   amc=amc, snr_db=snr_db, carrier_hint_hz=carrier,
                                   rolloff=float(opt["rolloff"]), max_hypotheses=5,
                                   with_fec=bool(opt.get("run_fec")),
                                   max_fec_bits=min(int(opt["max_fec_bits"]), 6000),
                                   cancelled=cancelled)
        hyp = out["hypotheses"]
        if hyp.get("best"):
            out["best_hypothesis"] = {
                "modulation": hyp["best"]["modulation"],
                "symbol_rate_hz": hyp["best"]["symbol_rate_hz"],
                "score": hyp["best"]["score"], "label": hyp["best"]["label"],
                "evm_percent": hyp["best"].get("evm_percent"),
                "ber_estimate": hyp["best"].get("ber_estimate"),
                "notes": hyp.get("notes", []),
            }

    # ---- parameters, quality and evidence graph --------------------------------------
    out["params"] = _parameter_block(out, spec, pr, amc, dm, fs, fs_known)
    out["quality"] = _quality_block(out, spec, dm)
    out["confidence"] = _confidence_summary(out)
    out["evidence_graph"] = _evidence_graph(out, region, modulation, rs)
    # NOTE: the full result is kept (bits, LLRs, symbol samples, chart arrays) so the FEC,
    # interleaving, correlation and bitstream endpoints can work on the *same* demodulated stream
    # without re-running the demodulator.  `to_jsonable` only guarantees JSON safety; the API layer
    # trims the payload for the browser (see backend/app/views.py) and the cache file is gzipped.
    out = to_jsonable(out)
    out["ok"] = True
    return out


def _trim_amc(amc: dict) -> dict:
    a = dict(amc or {})
    a.pop("features", None)
    return a


def _trim_constellation(cv: dict) -> dict:
    c = dict(cv or {})
    return c


def _trim_fec(res: dict) -> dict:
    r = dict(res or {})
    return r


def _trim_inter(res: dict) -> dict:
    r = dict(res or {})
    # the raw per-block metrics of every candidate are large; keep the essentials
    if r.get("decode_test"):
        r["decode_test"] = {k: v for k, v in r["decode_test"].items() if k != "candidates"} | {
            "candidates": [{k: v for k, v in c.items() if k != "params_list"}
                           for c in (r["decode_test"].get("candidates") or [])][:8]}
    return r


def _trim_corr(res: dict) -> dict:
    r = dict(res or {})
    return r


def _parameter_block(out: dict, spec: dict, pr: dict, amc: dict, dm: dict, fs: float,
                     fs_known: bool) -> dict:
    """The parameter table: value / confidence / method or 'unable to estimate'."""
    mod = out.get("modulation") or {}
    demod = out.get("demodulation") or {}
    blocks = {
        "record": [
            param("Sample rate", fs if fs_known else None, "Hz",
                  status="ok" if fs_known else "unable", confidence=1.0 if fs_known else None,
                  method="WAV header" if fs_known else "unknown - no metadata and no reference in "
                                                         "the data",
                  evidence=([f"{fs:,.0f} samples/s"] if fs_known else
                            ["the capture has no sample-rate metadata; all frequencies are "
                             "expressed as fractions of the sample rate"]),
                  limitations=[] if fs_known else
                  ["frequencies and symbol rates are reported in cycles/sample (x fs) because the "
                   "sample rate is unknown"]),
            param("Duration", out.get("duration_s"), "s", confidence=1.0,
                  method="number of samples / sample rate" if fs_known else "unknown sample rate",
                  status="ok" if fs_known else "unable"),
            param("Samples", out.get("n_samples"), None, confidence=1.0, method="file size / item size"),
        ],
        "spectrum": [v for v in (spec.get("params") or {}).values()],
        "modulation": (mod.get("params") or []),
        "demodulation": [param("Measured EVM", demod.get("quality", {}).get("evm_percent"), "%",
                               confidence=(0.85 if demod.get("quality", {}).get("evm_percent") is not None
                                           else None),
                               status="ok" if demod.get("quality", {}).get("evm_percent") is not None
                               else "unable",
                               method="RMS error to the ideal constellation after gain/phase fit",
                               evidence=[f"{demod.get('n_symbols')} symbols demodulated"]
                               if demod.get("n_symbols") else [])] if demod.get("ok") else
                          [unavailable("Measured EVM", "demodulation did not complete for this "
                                                       "emission")],
    }
    sr = (pr or {}).get("symbol_rate") or {}
    if isinstance(sr, dict) and sr.get("primary"):
        blocks["record"].append(sr["primary"])
    if out.get("symbol_rate_hz") and out.get("symbol_rate_note"):
        blocks["record"].append(param(
            "Symbol rate (refined by the demodulator)", round(float(out["symbol_rate_hz"]), 3), "Hz",
            confidence=0.85, method="timing-drift fit across the record (per-segment sampling phase "
                                   "fitted with a straight line)",
            evidence=[out["symbol_rate_note"]],
            limitations=["a rate measured at the wrong modulation order will lock onto a harmonic or "
                         "sub-multiple of the true symbol rate"]))
    return blocks


def _quality_block(out: dict, spec: dict, dm: dict) -> dict:
    demod = out.get("demodulation") or {}
    ok_demod = bool(demod.get("ok"))
    ber = (demod.get("ber_estimate") or {}).get("ber_estimate")
    return {
        "snr_db": spec.get("snr_db"), "evm_percent": (demod.get("quality") or {}).get("evm_percent"),
        "estimated_ber": ber, "bit_quality": demod.get("bit_quality"),
        "demodulation_ok": ok_demod,
        "summary": _quality_words(spec, demod),
        "stages_ok": sum(1 for s in demod.get("stages", []) if s.get("status") in ("ok", "accepted")),
        "stages_total": len(demod.get("stages", [])),
    }


def _quality_words(spec: dict, demod: dict) -> str:
    if not spec.get("ok"):
        return "the record could not be characterised spectrally"
    snr = spec.get("snr_db")
    if not demod.get("ok"):
        return ("the emission was characterised spectrally"
                + (f" at {snr:.1f} dB SNR" if snr is not None else "")
                + ", but demodulation did not complete - the parameter set is therefore partial")
    evm = (demod.get("quality") or {}).get("evm_percent")
    return (f"spectral SNR {snr:.1f} dB; demodulated {demod.get('n_symbols')} symbols with "
            f"{evm:.2f} % EVM" if evm is not None else
            f"spectral SNR {snr:.1f} dB; demodulation completed")


def _confidence_summary(out: dict) -> dict:
    mod = out.get("modulation") or {}
    spec = out.get("spectrum") or {}
    hyp = (out.get("hypotheses") or {}).get("best") or {}
    fec = (out.get("fec") or {}).get("best")
    inter = (out.get("interleaving") or {}).get("best")
    items = [
        {"name": "Signal presence", "confidence": clamp01(1.0 - (0.0 if spec.get("ok") else 1.0)),
         "basis": "CFAR detection and spectral characterisation"},
        {"name": "Parameter estimates", "confidence": clamp01((spec.get("snr_db") or 0) / 25.0),
         "basis": "SNR-dependent reliability of the estimators"},
        {"name": "Modulation", "confidence": float(mod.get("confidence") or 0.0),
         "basis": "hybrid DSP + evidence + ML classification"},
        {"name": "Demodulation", "confidence": clamp01(1.0 - ((out.get("quality") or {})
                                                              .get("evm_percent") or 100.0) / 40.0),
         "basis": "measured EVM against the hypothesis constellation"},
        {"name": "Hypothesis ranking", "confidence": float(hyp.get("score") or 0.0),
         "basis": "measurement-weighted multi-hypothesis score"},
        {"name": "FEC", "confidence": float((fec or {}).get("confidence") or 0.0),
         "basis": "evidence test per FEC family (0 = no claim)"},
        {"name": "Interleaving", "confidence": float((inter or {}).get("confidence") or 0.0),
         "basis": "dispersion test + de-interleave/decode improvement"},
    ]
    return {"items": items,
            "overall": float(np.mean([i["confidence"] for i in items])) if items else 0.0,
            "method": "each entry is the confidence of one automatic result; 0 means the platform "
                      "declined to make a claim"}


def _region_detail(region: dict) -> str:
    snr = region.get("snr_db")
    parts = []
    if region.get("f_lo_hz") is not None and region.get("f_hi_hz") is not None:
        parts.append(f"{float(region['f_lo_hz']):,.0f} .. {float(region['f_hi_hz']):,.0f} Hz")
    if snr is not None:
        parts.append(f"SNR {float(snr):.1f} dB")
    if region.get("duty_cycle") is not None:
        parts.append(f"duty {float(region['duty_cycle']) * 100:.0f} %")
    if region.get("id"):
        parts.insert(0, f"emission {region['id']}")
    return ", ".join(parts) if parts else "detected emission"


def _evidence_graph(out: dict, region: dict | None, modulation: str | None, rs: float | None) -> dict:
    """The ordered evidence chain shown in the UI (IQ -> ... -> bitstream)."""
    spec = out.get("spectrum") or {}
    demod = out.get("demodulation") or {}
    amc = out.get("modulation") or {}
    fec = (out.get("fec") or {}).get("best")
    inter = (out.get("interleaving") or {}).get("best")
    bits = out.get("bitstream") or {}
    nodes = [
        {"id": "iq", "label": "IQ / WAV record", "status": "ok",
         "detail": f"{out.get('n_samples'):,} samples at "
                   f"{out.get('fs'):,.0f} Hz" + ("" if out.get("fs_known", True)
                                                 else " (sample rate unknown)")},
        {"id": "spectrum", "label": "Spectrum / PSD", "status": "ok" if spec.get("ok") else "failed",
         "detail": (f"noise floor {spec.get('noise_floor_db'):.1f} dB, "
                    f"occupied bandwidth {human_si(spec.get('obw_99_hz'), 'Hz')}, "
                    f"SNR {spec.get('snr_db'):.1f} dB" if spec.get("ok") and spec.get("snr_db")
                    else spec.get("message", "not available"))},
        {"id": "spectrogram", "label": "Spectrogram / waterfall", "status": "ok",
         "detail": f"STFT {out.get('spectrogram', {}).get('nperseg', 'n/a')}-point, "
                   f"{out.get('spectrogram', {}).get('n_frames', 'n/a')} frames"},
        {"id": "candidate", "label": "Emission candidate",
         "status": "ok" if region else "context",
         "detail": (_region_detail(region) if region else "the whole capture")},
        {"id": "modulation", "label": "Modulation hypothesis", "status": "ok" if modulation else "failed",
         "detail": (f"{modulation} (confidence {amc.get('confidence', 0):.2f}), "
                    f"{len(amc.get('candidates') or [])} ranked alternatives"
                    if modulation else "no modulation candidate")},
        {"id": "timing", "label": "Symbol rate / timing", "status": "ok" if rs else "failed",
         "detail": f"{rs:,.1f} sym/s" if rs else "symbol rate could not be estimated"},
        {"id": "demod", "label": "Demodulation", "status": "ok" if demod.get("ok") else "failed",
         "detail": (f"{demod.get('n_symbols')} symbols, EVM "
                    f"{(demod.get('quality') or {}).get('evm_percent', float('nan')):.2f} %"
                    if demod.get("ok") else demod.get("message", "failed"))},
        {"id": "fec", "label": "FEC hypothesis",
         "status": "ok" if fec and fec.get("confidence", 0) >= 0.2 else "none",
         "detail": (f"{fec['hypothesis']} (confidence {fec['confidence']:.2f})" if fec and
                    fec.get("confidence", 0) >= 0.2 else
                    "no FEC family passed its evidence test")},
        {"id": "interleave", "label": "Interleaving hypothesis",
         "status": "ok" if inter and inter.get("confidence", 0) >= 0.2 else "none",
         "detail": (f"{inter['hypothesis']} (confidence {inter['confidence']:.2f})" if inter and
                    inter.get("confidence", 0) >= 0.2 else "no interleaver evidenced")},
        {"id": "bitstream", "label": "Bitstream", "status": "ok" if bits.get("n_bits") else "failed",
         "detail": (f"{bits.get('n_bits', 0):,} bits, entropy "
                    f"{bits.get('bit_entropy_per_bit')} bit/bit" if bits.get("n_bits") else
                    "no bits available")},
    ]
    return {"nodes": nodes,
            "note": "each node is a step whose output feeds the next; a 'none' node means the "
                    "platform declined to make a claim at that step"}


# ---------------------------------------------------------------------------------------
# record-level analysis
# ---------------------------------------------------------------------------------------
def analyse_record(x: np.ndarray, fs: float, file_info: dict | None = None,
                   options: dict | None = None, progress=None, cancelled=None,
                   fs_known: bool = True) -> dict:
    """Full record analysis: preprocessing, spectrum, waterfall, detection and one emission."""
    opt = dict(DEFAULT_OPTIONS)
    opt.update(options or {})
    st = StageLog()
    x = np.asarray(x)
    result: dict = {"ok": False, "options": {k: v for k, v in opt.items() if k != "preprocess_steps"},
                    "file": file_info or {}, "notes": [], "warnings": [],
                    "fs": float(fs), "fs_known": bool(fs_known), "n_samples": int(x.size)}
    if x.size < 256:
        result["message"] = (f"the record holds only {x.size} samples; at least 256 are required "
                             "for a meaningful analysis")
        st.add("load", "failed", result["message"])
        result["stages"] = st.summary()
        return result
    if np.allclose(np.abs(x), 0.0):
        result["message"] = "the record is all zeros (silent or empty capture)"
        st.add("load", "failed", result["message"])
        result["stages"] = st.summary()
        return result

    # ---- preprocessing ---------------------------------------------------------------
    if opt.get("preprocess"):
        prep = st.run("preprocessing", pre_mod.run_pipeline, x, fs, opt.get("preprocess_steps"))
        result["preprocessing"] = {
            "ok": bool(prep.get("ok", True)), "metrics_before": prep.get("metrics_before"),
            "metrics_after": prep.get("metrics_after"), "comparison": prep.get("comparison"),
            "log": prep.get("log"), "steps": prep.get("steps"), "flags": prep.get("flags"),
            "fs_out": prep.get("fs_out"), "message": prep.get("message"),
        }
        y = np.asarray(prep.get("samples")) if prep.get("samples") is not None else x
        fs_work = float(prep.get("fs_out") or fs)
    else:
        result["preprocessing"] = {"ok": True, "log": [], "steps": [],
                                   "note": "preprocessing disabled by the request options"}
        y, fs_work = x, fs
    result["analysis_fs"] = fs_work

    if cancelled and cancelled():
        result.update({"ok": False, "cancelled": True,
                       "message": "analysis cancelled by the client"})
        result["stages"] = st.summary()
        return result

    # ---- spectrogram + detection ------------------------------------------------------
    if opt.get("spectrogram"):
        pw = st.run("spectrogram", wf.waterfall_payload, y, fs_work, opt["nperseg"], 0.75, "hann",
                    opt["waterfall_max_w"], opt["waterfall_max_h"], "max")
        result["spectrogram"] = pw if pw.get("ok") else {"ok": False, "message": pw.get("message")}
    # record-level spectrum (before any emission is selected) - used by the explorer and by the
    # "whole capture" view in the report
    rec_spec = st.run("record spectrum", spec_mod.analyse_spectrum, y, fs_work, opt["nperseg"])
    if rec_spec.get("ok"):
        arrays = _spectrum_arrays(rec_spec)
        result["record_spectrum"] = {
            "no_signal": bool(rec_spec.get("no_signal", False)),
            "noise_floor_db": rec_spec.get("noise_floor_db"), "peak_db": rec_spec.get("peak_db"),
            "peak_frequency_hz": rec_spec.get("peak_frequency_hz"),
            "center_frequency_hz": rec_spec.get("center_frequency_hz"),
            "obw_99_hz": rec_spec.get("obw_99_hz"), "snr_db": rec_spec.get("snr_db"),
            "threshold_db": rec_spec.get("threshold_db"), "df_hz": rec_spec.get("df_hz"),
            "nperseg": rec_spec.get("nperseg"), "window": rec_spec.get("window"),
            "n_segments": rec_spec.get("n_segments"), "estimator": rec_spec.get("estimator"),
            "emissions": rec_spec.get("emissions", []),
            "occupied_fraction": rec_spec.get("occupied_fraction"),
            "line_like": rec_spec.get("line_like"), "flat_top": rec_spec.get("flat_top"),
            "numerical_floor": rec_spec.get("numerical_floor"),
            "message": rec_spec.get("message"),
            "arrays": arrays,
        }
    detection = None
    if opt.get("detect"):
        def _prog(frac, text):
            if progress:
                progress(0.05 + 0.25 * float(frac), text)
        detection = st.run("multi-signal detection", detect_mod.detect_signals, y, fs_work,
                           nperseg=opt["nperseg"], overlap=0.75, window="hann",
                           snr_threshold_db=6.0, min_cells=4,
                           max_signals=int(opt["max_detect_signals"]), merge_gap_bins=2,
                           p_fa=1e-3, cancelled=cancelled, progress=_prog)
        result["detection"] = detection
    if not detection or not detection.get("ok"):
        detection = {"ok": False, "signals": [], "notes": [str((detection or {}).get("message")
                                                              or "detection did not run")]}
        result["detection"] = detection

    if cancelled and cancelled():
        result.update({"ok": False, "cancelled": True,
                       "message": "analysis cancelled by the client"})
        result["stages"] = st.summary()
        return result

    # ---- choose the emission to analyse ------------------------------------------------
    signals = detection.get("signals") or []
    chosen = None
    if opt.get("region"):
        r = opt["region"]
        chosen = {"f_lo_hz": r.get("f_lo_hz"), "f_hi_hz": r.get("f_hi_hz"),
                  "t0_s": r.get("t0_s"), "t1_s": r.get("t1_s"), "manual": True,
                  "center_frequency_hz": (float(r.get("f_lo_hz") or 0) +
                                          float(r.get("f_hi_hz") or 0)) / 2.0 if
                  r.get("f_lo_hz") is not None else None}
    elif signals:
        if opt.get("signal_id"):
            chosen = next((s for s in signals if s.get("id") == opt["signal_id"]), None)
        chosen = chosen or detect_mod.rank_signals(signals)[0]
    # ---- extract the segment -----------------------------------------------------------
    segmentation = None
    seg_x, seg_fs = y, fs_work
    if chosen:
        f_lo, f_hi = chosen.get("f_lo_hz"), chosen.get("f_hi_hz")
        t0, t1 = chosen.get("t0_s"), chosen.get("t1_s")
        if f_lo is not None and f_hi is not None and not np.iscomplexobj(y) is False:
            pad = 0.60 * max(f_hi - f_lo, 1.0)
            segmentation = st.run("band extraction", wf.extract_band, y, fs_work,
                                  float(f_lo) - pad, float(f_hi) + pad, t0_s=t0, t1_s=t1)
            if segmentation.get("ok"):
                seg_x, seg_fs = segmentation["samples"], segmentation["fs"]
                result["segmentation"] = {k: v for k, v in segmentation.items() if k != "samples"}
                result["segment_center_hz"] = segmentation.get("f_center_hz")
            else:
                result["warnings"].append(f"band extraction failed: {segmentation.get('message')}")
    result["selected_signal"] = chosen
    if progress:
        progress(0.35, "analysing the selected emission")

    # ---- per-emission analysis ----------------------------------------------------------
    region_block = None
    if chosen:
        region_block = {k: chosen.get(k) for k in ("f_lo_hz", "f_hi_hz", "t0_s", "t1_s",
                                                   "snr_db", "bandwidth_hz", "confidence",
                                                   "center_frequency_hz", "id", "index")}
    band_extracted = bool(segmentation and segmentation.get("ok"))
    if band_extracted and region_block is not None:
        # the segment has been shifted to baseband: the original RF band edges no longer apply, but
        # the frequency offset of the emission inside the extracted segment does
        region_block = dict(region_block)
        region_block["f_lo_hz"] = None
        region_block["f_hi_hz"] = None
        region_block["center_frequency_hz"] = segmentation.get("f_center_hz")
    n0_hint = None
    n0_src = None
    if band_extracted and rec_spec.get("ok") and rec_spec.get("noise_density"):
        n0_hint = float(rec_spec["noise_density"])
        n0_src = ("carried over from the unfiltered record: a band-pass filter leaves the in-band "
                  "noise density unchanged, so measuring it again inside the filtered band would "
                  "overstate the SNR")
    sig = analyse_signal(seg_x, seg_fs, region=region_block, options=opt, stage=st,
                         progress=progress, cancelled=cancelled, fs_known=fs_known,
                         band_extracted=band_extracted, noise_density_hint=n0_hint,
                         noise_source=n0_src)
    result["signal"] = sig
    if segmentation and segmentation.get("ok"):
        sig["segment_center_hz"] = segmentation.get("f_center_hz")
        sig["segment_bandwidth_hz"] = segmentation.get("bw_hz")

    # ---- record-level conclusion ---------------------------------------------------------
    n_sig = len(signals)
    if n_sig == 0:
        result["notes"].append("no emission was detected above the noise floor: nothing is "
                               "reported beyond the characterisation of the noise itself")
    elif n_sig > 1:
        result["notes"].append(f"{n_sig} emissions were detected; the strongest/ranked one is "
                               "analysed in detail and every emission is listed with its own "
                               "measured parameters")
    result["ok"] = True
    result["signal_details"] = sig
    del result["signal_details"]
    result["stages"] = st.summary()
    result["confidence"] = sig.get("confidence") if isinstance(sig, dict) else None
    result["evidence_graph"] = sig.get("evidence_graph") if isinstance(sig, dict) else None
    result["quality"] = sig.get("quality") if isinstance(sig, dict) else None
    if progress:
        progress(0.95, "packaging the result")
    return to_jsonable(result)


# ---------------------------------------------------------------------------------------
# focused requests (used by the API routes)
# ---------------------------------------------------------------------------------------
def _resolve_input(x, fs, region=None, options=None):
    opt = dict(DEFAULT_OPTIONS)
    opt.update(options or {})
    y = np.asarray(x)
    fs_use = float(fs)
    seg = None
    if region and region.get("f_lo_hz") is not None and region.get("f_hi_hz") is not None:
        seg = wf.extract_band(y, fs_use, float(region["f_lo_hz"]), float(region["f_hi_hz"]),
                              t0_s=region.get("t0_s"), t1_s=region.get("t1_s"))
        if seg.get("ok"):
            return seg["samples"], seg["fs"], seg
    return y, fs_use, seg


def demodulate_request(x, fs, modulation=None, symbol_rate=None, rolloff=0.35, region=None,
                       options=None, stage=None) -> dict:
    """Explicit demodulation request (user-chosen modulation and/or symbol rate)."""
    st = stage or StageLog()
    opt = dict(DEFAULT_OPTIONS)
    opt.update(options or {})
    y, fs_use, seg = _resolve_input(x, fs, region, opt)
    spec = spec_mod.analyse_spectrum(y, fs_use)
    snr = spec.get("snr_db") if spec.get("ok") else None
    obw = spec.get("obw_99_hz") if spec.get("ok") else None
    amc = None
    rs = float(symbol_rate) if symbol_rate else None
    mod = modulation
    if not mod or not rs:
        if opt.get("blind", False) or not mod:
            amc = st.run("modulation classification", mod_mod.classify, y, fs_use, rs=rs,
                         obw_hz=obw, snr_db=snr, rolloff=float(rolloff))
            mod = mod or (amc.get("primary") if amc else None)
        if not rs:
            pr = st.run("symbol-rate estimation", params_mod.estimate_symbol_rate, y, fs_use,
                        obw_hz=obw, snr_db=snr)
            rs = ((pr.get("primary") or {}).get("symbol_rate_hz")
                  if isinstance(pr, dict) and pr.get("primary") else None)
    if not mod:
        return {"ok": False, "message": "no modulation could be determined; supply one explicitly "
                                        "or improve the signal quality"}
    dm = st.run(f"demodulation ({mod})", demod_mod.demodulate, y, fs_use, mod, rs,
                rolloff=float(rolloff), band_hint_hz=spec.get("center_frequency_hz"))
    out = {"ok": bool(dm.get("ok")), "modulation": mod, "symbol_rate_hz": rs,
           "modulation_auto": amc is not None, "snr_db": snr, "spectrum": {
               "center_frequency_hz": spec.get("center_frequency_hz"),
               "obw_99_hz": obw, "noise_floor_db": spec.get("noise_floor_db")},
           "segmentation": ({k: v for k, v in seg.items() if k != "samples"} if seg else None),
           "demodulation": {k: v for k, v in dm.items() if k not in ("symbols", "bits", "llrs")},
           "stages": st.summary()}
    if dm.get("ok"):
        out["symbols_preview"] = np.asarray(dm["symbols"])[:2000].tolist() \
            if dm.get("symbols") is not None else None
        out["bits_preview"] = "".join(str(int(b)) for b in np.asarray(dm["bits"])[:512]) \
            if dm.get("bits") is not None else None
        out["constellation"] = mod_mod.constellation_view(y, fs_use, mod, rs or 1.0,
                                                          rolloff=float(rolloff), view="symbols")
        out["eye"] = mod_mod.eye_diagram(y, fs_use, rs, rolloff=float(rolloff))
    if options and options.get("run_hypotheses", False):
        out["hypotheses"] = analyse_hypotheses(y, fs_use, amc=amc, snr_db=snr,
                                               band_hint_hz=spec.get("center_frequency_hz"))
    return to_jsonable(out)


def fec_request(bits, soft=None, max_bits=12000, cancelled=None) -> dict:
    res = fec_mod.analyse_fec(np.asarray(bits, dtype=np.int8), soft=soft, max_bits=int(max_bits),
                              cancelled=cancelled)
    res["param"] = fec_mod.fec_param(res)
    return to_jsonable(res)


def interleaving_request(bits, soft=None, fec_result=None, max_bits=8000, cancelled=None) -> dict:
    res = inter_mod.analyse_interleaving(soft=soft, bits=bits, fec_result=fec_result,
                                         max_bits=int(max_bits), cancelled=cancelled)
    res["param"] = inter_mod.interleaving_param(res)
    return to_jsonable(res)


def correlate_request(bits, patterns=None, max_errors=0, auto=True) -> dict:
    parsed = []
    for p in (patterns or []):
        q = corr_mod.parse_pattern(p.get("value"), kind=p.get("kind", "auto"))
        if q.get("ok"):
            q["max_errors"] = int(p.get("max_errors", max_errors) or 0)
            parsed.append(q)
    res = corr_mod.analyse_correlation(np.asarray(bits, dtype=np.uint8), patterns=parsed,
                                       max_errors=int(max_errors), auto=auto)
    res["params"] = corr_mod.correlation_params(res)
    return to_jsonable(res)


def analyse_region(x, fs, region: dict, options=None, progress=None, cancelled=None) -> dict:
    """Analyse a user-selected region of the record (click-to-analyze)."""
    opt = dict(DEFAULT_OPTIONS)
    opt.update(options or {})
    opt["region"] = region
    return analyse_record(x, fs, file_info=None, options=opt, progress=progress,
                          cancelled=cancelled)
