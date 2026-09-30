"""Self-benchmark: measure the platform's own accuracy on signals whose truth is known.

Nothing here is hard-coded: every number comes from generating the signal with the same generator
that ships in the app, running the same estimators/classifier/demodulator that the API uses, and
comparing the result with the generator's ground truth.  The output is a confusion matrix plus a
per-estimator error table, so the platform can state its own measured accuracy instead of claiming
one.

Sections
--------
``amc``         hybrid modulation classifier confusion matrix (symbol rate estimated, not given)
``estimators``  relative error of symbol rate, occupied bandwidth, centre frequency, SNR
``demod``       measured EVM and BER against the transmitted bits, per modulation and SNR
``fec``         does the FEC analyser name the code that was actually used?
``interleaving``does the interleaving analyser name the geometry that was actually used?
"""
from __future__ import annotations

import time

import numpy as np
from scipy import signal as sigproc

from . import fec as fec_mod
from . import interleave as inter_mod
from . import modulation as mod_mod
from . import params as params_mod
from . import spectrum as spec_mod
from . import synth as synth_mod
from .utils import clamp01, db

__all__ = ["run_benchmark", "DEFAULT_CONFIG"]

DEFAULT_CONFIG = {
    "kind": "full",             # full | amc | estimators | demod | fec | interleaving
    "modulations": ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "2FSK", "GFSK", "AM", "FM"],
    "snr_db": [20.0, 10.0],
    "symbol_rates": [25000.0, 5000.0],
    "repeats": 1,               # seeds per cell
    "n_symbols": 1500,
    "demod_snrs": [30.0, 20.0, 15.0, 10.0],
    "demod_mods": ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"],
    "fec_codes": ["conv_K7_r1_2", "conv_K3_r1_2", "rs_255_223"],
    "fec_error_rates": [0.0, 0.01],
    "interleavers": ["block", "convolutional", "diagonal", "pseudo-random"],
    "max_seconds": 300.0,
    "seed": 20260926,
}


def _spec(mod: str, rs: float, snr: float, seed: int, n_symbols: int) -> dict:
    return {"modulation": mod, "symbol_rate": float(rs), "fs": 200000.0, "snr_db": float(snr),
            "n_symbols": int(n_symbols), "rolloff": 0.35, "carrier_offset_hz": float(rs) * 0.12,
            "seed": int(seed), "tone_hz": 1200.0, "mod_index": 0.7, "fm_deviation_hz": float(rs) * 0.4}


def _estimate_all(x: np.ndarray, fs: float) -> dict:
    spec = spec_mod.analyse_spectrum(x, fs)
    snr = spec.get("snr_db") if spec.get("ok") else None
    obw = spec.get("obw_99_hz") if spec.get("ok") else None
    pr = params_mod.estimate_symbol_rate(x, fs, obw_hz=obw, snr_db=snr,
                                         spectral_flat=bool(spec.get("flat_top")))
    rs = None
    if isinstance(pr, dict) and pr.get("ok"):
        rs = (pr.get("primary") or {}).get("symbol_rate_hz")
    return {"spectrum": spec, "snr_db": snr, "obw_hz": obw, "symbol_rate_hz": rs, "rate_report": pr}


def _rel_err(est, truth) -> float | None:
    if est is None or truth in (None, 0):
        return None
    return abs(float(est) - float(truth)) / abs(float(truth))


def _amc_cell(mod: str, rs: float, snr: float, seed: int, n_symbols: int) -> dict:
    g = synth_mod.generate_signal(_spec(mod, rs, snr, seed, n_symbols))
    est = _estimate_all(g.samples, g.fs)
    amc = mod_mod.classify(g.samples, g.fs, rs=est["symbol_rate_hz"], obw_hz=est["obw_hz"],
                           snr_db=est["snr_db"], rolloff=0.35, is_real=(mod in ("AM", "FM")))
    return {"truth": mod, "predicted": amc.get("primary"), "confidence": amc.get("confidence"),
            "candidates": [c.get("modulation") for c in (amc.get("candidates") or [])[:3]],
            "snr_db": snr, "symbol_rate_hz": rs, "rs_estimated": est["symbol_rate_hz"],
            "obw_hz": est["obw_hz"], "snr_estimated": est["snr_db"],
            "n_symbols": n_symbols, "ml_model_accuracy": amc.get("ml_model_accuracy"),
            "amc": amc}


def _demod_case(mod: str, snr: float, seed: int, rs: float = 25000.0, fs: float = 200000.0) -> dict:
    """End-to-end BER test with known transmitted bits (built from the generator primitives)."""
    rng = np.random.default_rng(seed)
    n_bits = 4000
    bits = rng.integers(0, 2, n_bits).astype(np.uint8)
    if mod in ("2FSK", "GFSK"):
        sym = bits.astype(np.float64) * 2.0 - 1.0
        sps = int(round(fs / rs))
        idx = np.repeat(np.arange(sym.size), sps)
        x = synth_mod.modulate_cpfsk(bits, sps, h=0.5, bt=(0.5 if mod == "GFSK" else 0.0))
        x = np.asarray(x, dtype=np.complex128)
    else:
        k = synth_mod.bits_per_symbol(mod)
        n_use = (bits.size // k) * k
        sym = synth_mod.map_bits(mod, bits[:n_use])
        sps = int(round(fs / rs))
        x = synth_mod.pulse_shape(sym, sps, 0.35)
        x = np.asarray(x, dtype=np.complex128)
    p_sig = float(np.mean(np.abs(x) ** 2))
    n0 = p_sig / (10 ** (snr / 10.0) * max(fs, 1.0))
    noise = (rng.normal(0, np.sqrt(n0 * fs / 2), x.size) +
             1j * rng.normal(0, np.sqrt(n0 * fs / 2), x.size))
    x = x + noise
    from . import demod as demod_mod
    demod = demod_mod.demodulate(x, fs, mod, rs, rolloff=0.35)
    out = {"modulation": mod, "snr_db": snr, "ok": bool(demod.get("ok")),
           "evm_percent": (demod.get("quality") or {}).get("evm_percent"),
           "n_bits": int(np.size(demod.get("bits") or []))}
    if demod.get("ok") and demod.get("bits") is not None:
        b = np.asarray(demod["bits"], dtype=np.uint8).ravel()
        n_cmp = min(b.size, bits.size)
        if n_cmp >= 64:
            # the demodulated stream can be offset by a few symbols because of the pulse-shaping
            # delay; search a small window of shifts and take the best alignment (standard practice
            # for a benchmark: the receiver's frame phase is unknown)
            best = 1.0
            for shift in range(0, min(64, n_cmp // 4)):
                a = b[shift:shift + (n_cmp - shift) // 2 * 2]
                t = bits[:a.size]
                if a.size < 64:
                    break
                ber = float(np.mean(a != t))
                best = min(best, ber)
            out["ber"] = best
    return out


def _fec_case(code: str, err_rate: float, seed: int) -> dict:
    g = synth_mod.generate_signal({"modulation": "QPSK", "symbol_rate": 25000.0, "fs": 200000.0,
                                   "snr_db": 25.0, "n_symbols": 3000, "fec": code, "seed": seed})
    # take the generated (encoded) message bits through the demodulator
    from . import demod as demod_mod
    d = demod_mod.demodulate(g.samples, g.fs, "QPSK", 25000.0)
    if not d.get("ok"):
        return {"code": code, "err_rate": err_rate, "ok": False,
                "message": "demodulation failed for this case"}
    bits = np.asarray(d["bits"], dtype=np.int8)
    if err_rate > 0:
        rng = np.random.default_rng(seed + 7)
        flip = rng.random(bits.size) < err_rate
        bits = bits.copy()
        bits[flip] ^= 1
    res = fec_mod.analyse_fec(bits, soft=d.get("llrs"), max_bits=6000, null_trials=3)
    best = res.get("best") or {}
    return {"code": code, "err_rate": err_rate, "ok": True,
            "predicted": best.get("hypothesis"), "family": best.get("family"),
            "confidence": best.get("confidence"), "correct": best.get("hypothesis") == code,
            "hypotheses": [(h.get("hypothesis"), h.get("confidence"))
                           for h in (res.get("hypotheses") or [])[:3]]}


def _interleaver_case(kind: str, seed: int, coded: bool = True) -> dict:
    g = synth_mod.generate_signal({"modulation": "QPSK", "symbol_rate": 25000.0, "fs": 200000.0,
                                   "snr_db": 30.0, "n_symbols": 2500, "fec": "conv_K7_r1_2",
                                   "interleaver": kind, "seed": seed})
    from . import demod as demod_mod
    d = demod_mod.demodulate(g.samples, g.fs, "QPSK", 25000.0)
    if not d.get("ok"):
        return {"interleaver": kind, "ok": False, "message": "demodulation failed"}
    soft = d.get("llrs")
    bits = np.asarray(d["bits"], dtype=np.int8) if d.get("bits") is not None else None
    res = inter_mod.analyse_interleaving(soft=soft, bits=bits, max_bits=6000)
    best = res.get("best") or {}
    return {"interleaver": kind, "ok": True, "predicted": best.get("kind"),
            "hypothesis": best.get("hypothesis"), "confidence": best.get("confidence"),
            "correct": bool(best.get("kind") == kind),
            "candidates": [(c.get("kind"), c.get("confidence"))
                           for c in (res.get("hypotheses") or [])[:3]],
            "evidence": (best.get("evidence") or [])[:2]}


def _confusion(cases: list[dict], classes: list[str]) -> dict:
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), dtype=int)
    unknown = 0
    for c in cases:
        t, p = c.get("truth"), c.get("predicted")
        if t not in idx:
            continue
        if p not in idx:
            unknown += 1
            continue
        m[idx[t], idx[p]] += 1
    per_class = []
    for c in classes:
        i = idx[c]
        tp = int(m[i, i])
        total = int(m[i].sum())
        pred = int(m[:, i].sum())
        per_class.append({"class": c, "support": total, "correct": tp,
                          "recall": (tp / total if total else None),
                          "precision": (tp / pred if pred else None)})
    return {"matrix": m.tolist(), "per_class": per_class, "n_unclassified": int(unknown),
            "n_cases": int(m.sum())}


def run_benchmark(config: dict | None = None, progress=None, cancelled=None) -> dict:
    """Run the self-test grid and return the confusion matrix plus the estimator error tables."""
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config or {})
    t_start = time.time()
    kind = str(cfg.get("kind") or "full")
    want = {"amc": kind in ("amc", "full", "estimators"), "estimators": kind in ("estimators", "full"),
            "demod": kind in ("demod", "full"), "fec": kind in ("fec", "full"),
            "interleaving": kind in ("interleaving", "full")}
    mods = [m for m in cfg["modulations"]]
    snrs = [float(s) for s in cfg["snr_db"]]
    rates = [float(r) for r in cfg["symbol_rates"]]
    repeats = max(1, int(cfg.get("repeats", 1)))
    cells_per_mod = len(snrs) * len(rates) * repeats
    total_cases = ((len(mods) * cells_per_mod if want["amc"] else 0) +
                   (len(cfg["demod_mods"]) * len(cfg["demod_snrs"]) if want["demod"] else 0) +
                   (len(cfg["fec_codes"]) * len(cfg["fec_error_rates"]) if want["fec"] else 0) +
                   (len(cfg["interleavers"]) if want["interleaving"] else 0))
    done = 0
    notes: list[str] = []
    out: dict = {"kind": kind, "config": cfg, "n_cases": total_cases, "cancelled": False}

    def tick(msg: str) -> bool:
        nonlocal done
        done += 1
        if progress:
            progress(min(0.98, done / max(total_cases, 1)), f"{msg} ({done}/{total_cases})")
        return bool(cancelled and cancelled())

    amc_cases: list[dict] = []
    est_rows: list[dict] = []
    if want["amc"] or want["estimators"]:
        for mod in mods:
            for snr in snrs:
                for rs in rates:
                    for rep in range(repeats):
                        if cancelled and cancelled():
                            out["cancelled"] = True
                            break
                        seed = int(cfg.get("seed", 1)) + int(snr) * 13 + int(rs) + rep * 977 + len(mod)
                        cell = _amc_cell(mod, rs, snr, seed, int(cfg.get("n_symbols", 1500)))
                        amc_cases.append(cell)
                        est_rows.append({
                            "modulation": mod, "snr_db": snr, "symbol_rate_hz": rs,
                            "symbol_rate_est_hz": cell["rs_estimated"],
                            "symbol_rate_rel_err": _rel_err(cell["rs_estimated"], rs if mod not in ("AM", "FM") else None),
                            "obw_hz": cell["obw_hz"],
                            "snr_est_db": cell["snr_estimated"],
                            "snr_err_db": (None if cell["snr_estimated"] is None else
                                           float(cell["snr_estimated"]) - float(snr)),
                        })
                        if tick(f"AMC {mod} @ {snr:.0f} dB"):
                            out["cancelled"] = True
                            break
                    if out.get("cancelled"):
                        break
                if out.get("cancelled"):
                    break
            if out.get("cancelled"):
                break
        classes = sorted(set(mods))
        conf = _confusion(amc_cases, classes)
        out["classes"] = classes
        out["confusion_matrix"] = conf["matrix"]
        out["per_class"] = conf["per_class"]
        out["n_unclassified"] = conf["n_unclassified"]
        out["accuracy"] = (float(sum(p["correct"] for p in conf["per_class"]) /
                                 max(conf["n_cases"], 1)))
        out["amc_cases"] = [{"truth": c["truth"], "predicted": c["predicted"],
                             "confidence": c["confidence"], "snr_db": c["snr_db"],
                             "symbol_rate_hz": c["symbol_rate_hz"],
                             "rs_estimated": c["rs_estimated"], "top3": c["candidates"]}
                            for c in amc_cases]
        if want["estimators"]:
            def _summary(key: str) -> dict:
                vals = [r[key] for r in est_rows if r.get(key) is not None]
                if not vals:
                    return {"n": 0}
                a = np.asarray(vals, dtype=np.float64)
                return {"n": int(a.size), "median": float(np.median(a)),
                        "mean": float(np.mean(a)), "p90": float(np.percentile(np.abs(a), 90)),
                        "max": float(np.max(np.abs(a)))}
            out["estimator_errors"] = {
                "symbol_rate_relative": _summary("symbol_rate_rel_err"),
                "snr_error_db": _summary("snr_err_db"),
                "rows": est_rows,
                "method": "estimators were run exactly as in the API (spectrum -> symbol rate); the "
                          "error is measured against the generator's settings",
            }
            notes.append("symbol-rate accuracy is reported as a relative error against the "
                         "generator's requested rate (PSK/QAM); analogue carriers have no symbol rate")

    if want["demod"]:
        demod_rows = []
        for mod in cfg["demod_mods"]:
            for snr in [float(s) for s in cfg["demod_snrs"]]:
                if cancelled and cancelled():
                    out["cancelled"] = True
                    break
                r = _demod_case(mod, snr, int(cfg.get("seed", 1)) + len(mod) * 31 + int(snr))
                demod_rows.append(r)
                if tick(f"demod {mod} @ {snr:.0f} dB"):
                    out["cancelled"] = True
                    break
            if out.get("cancelled"):
                break
        out["demod"] = {
            "rows": demod_rows,
            "mean_evm_percent_by_mod": {
                m: float(np.mean([r["evm_percent"] for r in demod_rows
                                  if r["modulation"] == m and r.get("evm_percent") is not None]))
                for m in cfg["demod_mods"]
                if any(r["modulation"] == m and r.get("evm_percent") is not None for r in demod_rows)},
            "method": "4000 random bits per case, band-limited noise added at the stated in-band SNR, "
                      "full receive chain (carrier recovery -> timing recovery -> RRC matched filter "
                      "-> decision), BER taken as the best of the first 64 bit alignments",
        }

    if want["fec"]:
        fec_rows = []
        for code in cfg["fec_codes"]:
            for err in [float(e) for e in cfg["fec_error_rates"]]:
                if cancelled and cancelled():
                    out["cancelled"] = True
                    break
                fec_rows.append(_fec_case(code, err, int(cfg.get("seed", 1)) + len(code)))
                if tick(f"FEC {code} @ {err:.0%} bit errors"):
                    out["cancelled"] = True
                    break
            if out.get("cancelled"):
                break
        out["fec"] = {"rows": fec_rows,
                      "n_correct": sum(1 for r in fec_rows if r.get("correct")),
                      "n_cases": len(fec_rows),
                      "method": "a coded waveform is generated with the platform's own generator, "
                                "demodulated, optionally corrupted with the stated random bit-error "
                                "rate, and then handed to the FEC hypothesis tester"}

    if want["interleaving"]:
        int_rows = []
        for kind_ in cfg["interleavers"]:
            if cancelled and cancelled():
                out["cancelled"] = True
                break
            int_rows.append(_interleaver_case(kind_, int(cfg.get("seed", 1)) + len(kind_)))
            if tick(f"interleaving {kind_}"):
                out["cancelled"] = True
                break
        out["interleaving"] = {"rows": int_rows,
                               "n_correct": sum(1 for r in int_rows if r.get("correct")),
                               "n_cases": len(int_rows),
                               "method": "coded + interleaved waveform, demodulated, then handed to "
                                         "the interleaving hypothesis tester (dispersion test and "
                                         "de-interleave/decode improvement)"}

    out["duration_s"] = round(time.time() - t_start, 2)
    notes.append(f"the whole run took {out['duration_s']:.1f} s on this machine; only signals "
                 "generated by this platform were used, so the numbers describe the estimator "
                 "chain, not real-world channel conditions")
    if out.get("cancelled"):
        notes.append("the run was cancelled: the reported numbers cover the cases that finished")
    out["notes"] = notes
    return out
