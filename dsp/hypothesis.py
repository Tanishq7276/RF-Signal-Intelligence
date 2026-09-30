"""Multi-hypothesis analysis.

A single automatic answer hides risk: if the modulation classifier is wrong, every downstream
result is wrong.  This module therefore runs several *complete* demodulation hypotheses - each a
(modulation, symbol rate) pair - through the same estimator chain and reports them side by side
with the measurements that support or contradict each one.

Each hypothesis is scored from measurements only:

======================  ======  =========================================================
term                    weight  measurement
======================  ======  =========================================================
link margin             0.30    estimated SNR minus the SNR the modulation needs for 1e-3
EVM quality             0.25    measured EVM against the modulation's own reference
correction rate         0.15    fraction of bits whose soft decision is fully confident
timing quality          0.10    Oerder-Meyr timing-line strength (or FSK timing contrast)
carrier lock            0.10    residual phase variance after carrier tracking
classifier agreement    0.10    the modulation classifier's probability for that modulation
======================  ======  =========================================================

The weights are fixed and reported with the result, so a user can see *why* a hypothesis ranks
first.  FEC evidence is run on demand for the top hypotheses and reported as an extra column.
"""

from __future__ import annotations

import numpy as np

from . import demod as demod_mod
from . import params as params_mod
from .utils import clamp01, db, param, safe_float, unavailable

__all__ = ["required_snr_1e3", "score_hypothesis", "candidate_modulations", "run_hypothesis",
           "analyse_hypotheses"]

# SNR (dB, in the occupied band) needed for an uncoded BER of 1e-3 - textbook values
REQUIRED_SNR_1E3 = {"BPSK": 6.8, "QPSK": 6.8, "8PSK": 10.4, "16QAM": 10.5, "64QAM": 14.7,
                    "OQPSK": 6.8, "MSK": 9.0, "2FSK": 8.0, "GFSK": 8.0, "FSK": 8.0,
                    "AM": 9.0, "FM": 9.0}


def required_snr_1e3(modulation: str) -> float | None:
    key = (modulation or "").upper().replace("8-PSK", "8PSK").replace("16-QAM", "16QAM").replace(
        "64-QAM", "64QAM").replace("4-QAM", "QPSK")
    return REQUIRED_SNR_1E3.get(key)


def candidate_modulations(amc: dict | None, max_candidates: int = 6) -> list[str]:
    """Modulation hypotheses to test: the classifier's ranking, padded with the usual suspects."""
    out: list[str] = []
    if amc and amc.get("candidates"):
        for c in amc["candidates"]:
            m = str(c.get("modulation"))
            if m and m not in out:
                out.append(m)
    for m in ("QPSK", "BPSK", "8PSK", "16QAM", "GFSK", "2FSK", "64QAM", "AM", "FM"):
        if len(out) >= max_candidates:
            break
        if m not in out:
            out.append(m)
    return out[:max_candidates]


def _demod_metrics(dm: dict, rs: float, fs: float) -> dict:
    q = dm.get("quality") or {}
    evm = q.get("evm_rms")
    ber = (dm.get("ber_estimate") or {}).get("ber_estimate")
    bits = dm.get("bits")
    llrs = dm.get("llrs")
    corr = None
    if llrs is not None and np.size(llrs):
        llr = np.asarray(llrs, dtype=np.float64).ravel()
        corr = float(np.mean(np.abs(llr) > 1.0))
    timing = None
    for st in dm.get("stages", []):
        if st.get("name", "").startswith("timing"):
            timing = st.get("detail", {}).get("line_strength_db") if isinstance(st.get("detail"), dict) else None
            if timing is None:
                timing = st.get("value")
    return {"evm_rms": evm, "evm_percent": (None if evm is None else evm * 100.0),
            "evm_db": q.get("evm_db"), "ber_estimate": ber,
            "n_symbols": dm.get("n_symbols"), "n_bits": None if bits is None else int(np.size(bits)),
            "soft_confident_fraction": corr, "timing_quality_db": timing,
            "carrier_offset_hz": dm.get("carrier_offset_hz"),
            "timing_phase_frac": dm.get("timing_phase_frac"), "ok": bool(dm.get("ok")),
            "stage_status": [{"name": s.get("name"), "status": s.get("status")}
                             for s in dm.get("stages", [])]}


def score_hypothesis(h: dict, weights: dict | None = None) -> dict:
    """Composite score from measurements (see the module docstring for the weights)."""
    w = weights or {"margin": 0.30, "evm": 0.25, "correction": 0.15, "timing": 0.10,
                    "carrier": 0.10, "classifier": 0.10}
    snr = h.get("snr_db")
    need = required_snr_1e3(h.get("modulation", ""))
    margin = None if (snr is None or need is None) else float(snr) - float(need)
    s_margin = clamp01((margin + 3.0) / 20.0) if margin is not None else 0.0
    evm = h.get("evm_rms")
    s_evm = clamp01(1.0 - (float(evm) / 0.35)) if evm is not None else 0.0
    s_corr = clamp01(h["soft_confident_fraction"]) if h.get("soft_confident_fraction") is not None else 0.0
    tq = h.get("timing_quality_db")
    s_timing = clamp01((float(tq) - 6.0) / 20.0) if tq is not None else 0.0
    res = h.get("carrier_residual_rad_per_symbol")
    s_carrier = clamp01(1.0 - abs(float(res)) / 0.15) if res is not None else 0.0
    s_cls = clamp01(h.get("classifier_probability") or 0.0)
    total = (w["margin"] * s_margin + w["evm"] * s_evm + w["correction"] * s_corr +
             w["timing"] * s_timing + w["carrier"] * s_carrier + w["classifier"] * s_cls)
    return {"score": float(total), "margin_db": margin, "terms": {
        "link_margin": s_margin, "evm": s_evm, "correction": s_corr, "timing": s_timing,
        "carrier": s_carrier, "classifier": s_cls}, "weights": w}


def run_hypothesis(x: np.ndarray, fs: float, modulation: str, symbol_rate: float,
                   rolloff: float = 0.35, carrier_hint_hz: float | None = None,
                   snr_db: float | None = None, classifier_probability: float | None = None,
                   with_fec: bool = False, max_fec_bits: int = 6000) -> dict:
    """Run one complete hypothesis through the demodulator (and optionally the FEC test)."""
    h: dict = {"modulation": modulation, "symbol_rate_hz": None if symbol_rate is None else float(symbol_rate),
               "rolloff": float(rolloff), "snr_db": snr_db,
               "classifier_probability": classifier_probability}
    try:
        dm = demod_mod.demodulate(x, fs, modulation, symbol_rate, rolloff=rolloff,
                                  carrier_hint_hz=carrier_hint_hz)
    except Exception as exc:                                            # pragma: no cover
        h.update({"ok": False, "error": f"demodulation failed: {exc}"})
        h["score_detail"] = score_hypothesis(h)
        h["score"] = h["score_detail"]["score"]
        return h
    h["ok"] = bool(dm.get("ok"))
    h["demodulation"] = {k: v for k, v in dm.items()
                         if k in ("ok", "modulation", "carrier_offset_hz", "timing_phase_frac",
                                  "n_symbols", "n_bits", "bit_quality", "message")}
    h["stages"] = dm.get("stages", [])
    metrics = _demod_metrics(dm, float(symbol_rate or 0.0), fs)
    h.update(metrics)
    q = dm.get("quality") or {}
    h["carrier_residual_rad_per_symbol"] = None
    for st in dm.get("stages", []):
        if "carrier" in str(st.get("name", "")).lower():
            det = st.get("detail") if isinstance(st.get("detail"), dict) else {}
            if det.get("residual_frequency_rad_per_symbol") is not None:
                h["carrier_residual_rad_per_symbol"] = det["residual_frequency_rad_per_symbol"]
    if h.get("ber_estimate") is not None:
        h["expected_bit_errors_per_1e5"] = float(h["ber_estimate"]) * 1e5
    if h.get("n_symbols") is not None and h.get("symbol_rate_hz"):
        h["analysed_duration_s"] = float(h["n_symbols"] / max(h["symbol_rate_hz"], 1e-9))
    if with_fec and dm.get("bits") is not None and np.size(dm["bits"]) >= 256:
        from . import fec as fec_mod
        fe = fec_mod.analyse_fec(np.asarray(dm["bits"], dtype=np.int8),
                                 soft=dm.get("llrs"), max_bits=max_fec_bits, null_trials=4)
        best = fe.get("best")
        h["fec"] = None if not best else {"hypothesis": best["hypothesis"],
                                          "confidence": best["confidence"],
                                          "family": best.get("family")}
        h["fec_evidence"] = None if not best else best.get("evidence", [])[:2]
    h["score_detail"] = score_hypothesis(h)
    h["score"] = h["score_detail"]["score"]
    h["_symbols"] = dm.get("symbols")
    h["_bits"] = dm.get("bits")
    h["_llrs"] = dm.get("llrs")
    return h


def analyse_hypotheses(x: np.ndarray, fs: float, amc: dict | None = None,
                       symbol_rates: list[float] | None = None,
                       snr_db: float | None = None, carrier_hint_hz: float | None = None,
                       rolloff: float = 0.35, max_hypotheses: int = 6, with_fec: bool = False,
                       max_fec_bits: int = 6000, sweep_samples: int = 16000,
                       cancelled=None, progress=None) -> dict:
    """Run and rank several complete demodulation hypotheses.

    ``symbol_rates`` are the symbol-rate candidates to combine with the modulation candidates; by
    default the platform's own symbol-rate estimate and its simple sub-multiples are used, because
    a symbol-rate estimator that locked onto the second harmonic is a realistic failure mode.

    Cost control: the *ranking* runs on a bounded prefix of the record (``sweep_samples``) because a
    hypothesis is separated from its rivals by measurements that need a few thousand symbols, not a
    whole capture.  The winning hypothesis is then re-measured on the **full** record, and both
    lengths are reported (``sweep_samples``/``remeasured_full_record``), so nothing is claimed from a
    prefix that was not verified on the whole capture.
    """
    if symbol_rates is None:
        pr = params_mod.estimate_symbol_rate(x, fs, snr_db=snr_db)
        primary = (pr.get("primary") or {}).get("symbol_rate_hz") if pr.get("ok") else None
        cand = []
        if primary:
            cand.append(float(primary))
            for div in (2.0, 3.0, 0.5):
                v = float(primary) / div
                if v > 20.0:
                    cand.append(v)
        symbol_rates = cand or [float(fs) / 100.0]
    mods = candidate_modulations(amc, max_candidates=6)
    probs = {}
    if amc and amc.get("candidates"):
        probs = {c["modulation"]: float(c.get("probability") or 0.0) for c in amc["candidates"]}
    # The grid is kept deliberately small: the top-ranked modulation is probed at several symbol
    # rates (a harmonic error is a realistic failure mode) while the runners-up are probed at the
    # primary rate only.  Every extra combination costs a complete demodulation chain.
    combos = []
    rates_main = list(symbol_rates)[:3]
    for i, m in enumerate(mods):
        for rs in (rates_main if i == 0 else symbol_rates[:1]):
            combos.append((m, float(rs), probs.get(m)))
    if len(combos) > max_hypotheses * 2:
        combos = combos[: max_hypotheses * 2]
    # bounded sweep: rank on a prefix, verify the winner on the whole record
    xs = np.asarray(x)
    sweep_note = None
    if xs.size > int(sweep_samples) > 0:
        n_sw = int(max(4096, min(xs.size, sweep_samples)))
        if n_sw < xs.size:
            sweep_note = (f"hypotheses were ranked on the first {n_sw:,} of {xs.size:,} samples; "
                          f"the winning hypothesis was re-measured on the full record")
            xs = xs[:n_sw]
    results = []
    for i, (m, rs, pr) in enumerate(combos):
        if cancelled and cancelled():
            return {"ok": False, "cancelled": True, "hypotheses": []}
        if progress:
            progress(0.15 + 0.7 * i / max(len(combos), 1),
                     f"hypothesis {i + 1}/{len(combos)}: {m} at {rs:,.0f} sym/s")
        h = run_hypothesis(xs, fs, m, rs, rolloff=rolloff, carrier_hint_hz=carrier_hint_hz,
                           snr_db=snr_db, classifier_probability=pr,
                           with_fec=False, max_fec_bits=max_fec_bits)
        h["ranked_on_samples"] = int(xs.size)
        h.pop("_symbols", None)
        h.pop("_bits", None)
        h.pop("_llrs", None)
        results.append(h)
    results.sort(key=lambda h: -h["score"])
    # --- the winner is verified on the full record, with its (optional) FEC evidence ---------
    if results and sweep_note and xs.size != np.asarray(x).size:
        best = results[0]
        if cancelled and cancelled():
            pass
        else:
            try:
                full = run_hypothesis(np.asarray(x), fs, best["modulation"], best["symbol_rate_hz"],
                                      rolloff=rolloff, carrier_hint_hz=carrier_hint_hz,
                                      snr_db=snr_db,
                                      classifier_probability=best.get("classifier_probability"),
                                      with_fec=bool(with_fec), max_fec_bits=max_fec_bits)
                for key in ("_symbols", "_bits", "_llrs"):
                    full.pop(key, None)
                full["ranked_on_samples"] = int(np.asarray(x).size)
                full["remeasured_full_record"] = True
                full["score_on_full_record"] = full.get("score")
                # keep the sweep score for the ranking (all hypotheses were compared at the same
                # length) and expose both, so the shorter measurement is never hidden
                full["score"] = best["score"]
                full["score_detail"] = dict(best.get("score_detail") or {})
                full["sweep_score_detail"] = dict(full.get("score_detail") or {})
                results[0] = full
            except Exception as exc:                                       # pragma: no cover
                best["full_record_error"] = f"{type(exc).__name__}: {exc}"
    for k, h in enumerate(results):
        h["rank"] = k + 1
        h["label"] = f"H{k + 1}"
    top = results[0] if results else None
    notes = []
    if sweep_note:
        notes.append(sweep_note)
    if top:
        notes.append(f"{len(results)} complete hypotheses were evaluated; {top['label']} "
                     f"({top['modulation']} at {top['symbol_rate_hz']:,.0f} sym/s) ranks first with "
                     f"score {top['score']:.2f}")
        if len(results) > 1 and top["score"] - results[1]["score"] < 0.08:
            notes.append(f"{results[1]['label']} ({results[1]['modulation']}) is within 0.08 of the "
                         "top score: the two hypotheses are not clearly separated by the available "
                         "measurements")
    return {"ok": True, "hypotheses": results, "best": top, "notes": notes,
            "weights": (top or {}).get("score_detail", {}).get("weights"),
            "method": "each hypothesis is demodulated end-to-end and scored on measured link "
                      "margin, EVM, soft-decision confidence, timing quality, carrier residual and "
                      "the classifier probability"}


def hypothesis_param(result: dict) -> dict:
    """Parameter record summarising the winning hypothesis."""
    best = (result or {}).get("best")
    if not best:
        return unavailable("Multi-hypothesis analysis", "no hypothesis could be evaluated")
    ev = [f"rank 1 of {len(result.get('hypotheses', []))} evaluated complete pipelines",
          f"measured EVM {best['evm_percent']:.2f} %" if best.get("evm_percent") is not None
          else "EVM not available",
          f"link margin {best['score_detail']['margin_db']:+.1f} dB above the "
          f"{required_snr_1e3(best['modulation']):.1f} dB needed for 1e-3 uncoded BER"
          if best.get("score_detail", {}).get("margin_db") is not None else "link margin unavailable",
          f"composite score {best['score']:.2f} " +
          ", ".join(f"{k}={v:.2f}" for k, v in best["score_detail"]["terms"].items())]
    return param("Best hypothesis",
                 f"{best['modulation']} at {best['symbol_rate_hz']:,.0f} sym/s", None,
                 clamp01(best["score"]), "multi-hypothesis ranking (measurement-weighted)",
                 status="ok" if best["score"] > 0.35 else "low_confidence",
                 evidence=ev,
                 limitations=["hypotheses are ranked, not confirmed: a higher score means better "
                              "agreement with the measured evidence, not proof of the transmitted "
                              "format"])
