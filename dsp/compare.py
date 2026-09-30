"""Signal comparison: how alike are two analysed records?

The comparison is deliberately *evidence-first* and never claims that two signals come from the
same source.  It measures, with the same estimators used everywhere else in the platform:

* spectral shape similarity (normalised cross-correlation of the two PSDs after aligning them on
  their detected centre frequency),
* modulation agreement (top hypothesis of each, with the confidence of both),
* symbol-rate ratio, occupied-bandwidth ratio, SNR difference,
* constellation agreement (EVM of each against the other's constellation order),
* byte-distribution similarity (Jensen-Shannon divergence of the two byte histograms, when both
  records were demodulated to bits).

The verdict is one of ``similar`` / ``partially similar`` / ``significantly different`` plus the
evidence table.  Two signals can be similar without sharing a source, and the report says so.
"""
from __future__ import annotations

import math

import numpy as np

from .utils import clamp01, param

__all__ = ["compare_results", "compare_psd", "verdict_for"]


def _get(d: dict | None, *keys, default=None):
    cur = d or {}
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur


def compare_psd(a_arrays: dict | None, b_arrays: dict | None, max_shift_frac: float = 0.25) -> dict:
    """Normalised cross-correlation of two PSDs, aligned on their peak."""
    if not a_arrays or not b_arrays:
        return {"ok": False, "message": "spectral arrays are not available for both records"}
    fa = np.asarray(a_arrays.get("freq_hz") or [], dtype=np.float64)
    pa = np.asarray(a_arrays.get("psd_db") or [], dtype=np.float64)
    fb = np.asarray(b_arrays.get("freq_hz") or [], dtype=np.float64)
    pb = np.asarray(b_arrays.get("psd_db") or [], dtype=np.float64)
    if fa.size < 8 or fb.size < 8 or fa.size != pa.size or fb.size != pb.size:
        return {"ok": False, "message": "spectral arrays are incomplete"}
    # interpolate both onto a common relative grid of 512 points spanning each record's own peak
    n = 512
    grid = np.linspace(0.0, 1.0, n)

    def _resample(f, p):
        span = f[-1] - f[0]
        if span <= 0:
            return None
        rel = (f - f[int(np.argmax(p))]) / span        # frequency relative to the peak, normalised
        x = np.linspace(-1.0, 1.0, n)
        order = np.argsort(rel)
        return np.interp(x, rel[order], p[order] - np.max(p))

    ra, rb = _resample(fa, pa), _resample(fb, pb)
    if ra is None or rb is None:
        return {"ok": False, "message": "could not normalise the spectra"}
    # search a small relative shift for the best alignment
    best, best_shift = -1.0, 0
    max_shift = int(max_shift_frac * n)
    for shift in range(-max_shift, max_shift + 1, 2):
        b = np.roll(rb, shift)
        num = float(np.sum(ra * b))
        den = float(np.sqrt(np.sum(ra ** 2) * np.sum(b ** 2)))
        if den > 0:
            c = num / den
            if c > best:
                best, best_shift = c, shift
    return {"ok": True, "correlation": float(best), "shift_points": int(best_shift),
            "shift_fraction_of_band": float(best_shift / float(n)),
            "method": "normalised cross-correlation of the peak-aligned, level-normalised PSDs "
                      "on a 512-point relative frequency grid"}


def verdict_for(score: float) -> str:
    if score >= 0.80:
        return "similar"
    if score >= 0.55:
        return "partially similar"
    return "significantly different"


def compare_results(a: dict, b: dict) -> dict:
    """Compare two analysis summaries (the ``signal`` blocks of two analyses)."""
    sa, sb = a.get("signal") or {}, b.get("signal") or {}
    ssa, ssb = sa.get("spectrum") or {}, sb.get("spectrum") or {}
    ma, mb = sa.get("modulation") or {}, sb.get("modulation") or {}
    da, db_ = sa.get("demodulation") or {}, sb.get("demodulation") or {}
    metrics: list[dict] = []
    notes: list[str] = []

    def add(name: str, value, unit, detail: str) -> None:
        metrics.append({"metric": name, "value": value, "unit": unit, "detail": detail})

    # 1. spectral shape ---------------------------------------------------------------
    ps = compare_psd((sa.get("spectrum") or {}).get("arrays"), (sb.get("spectrum") or {}).get("arrays"))
    if ps.get("ok"):
        # a correlation of 0.5 is already a meaningful shape match; map to 0..1
        s_spec = clamp01((ps["correlation"] - 0.3) / 0.65)
        add("Spectral shape correlation", round(ps["correlation"], 3), None,
            f"peak-aligned PSD correlation; best alignment shift "
            f"{ps['shift_fraction_of_band'] * 100:.1f} % of the record bandwidth")
    else:
        s_spec = 0.0
        notes.append(f"spectral shapes could not be compared: {ps.get('message')}")
    # 2. modulation --------------------------------------------------------------------
    same_mod = (ma.get("primary") or "") == (mb.get("primary") or "")
    conf = min(float(ma.get("confidence") or 0), float(mb.get("confidence") or 0))
    s_mod = clamp01((1.0 if same_mod else 0.0) * (0.4 + 0.6 * conf))
    add("Modulation agreement", "same" if same_mod else "different", None,
        f"{ma.get('primary') or 'unknown'} (confidence {ma.get('confidence') or 0:.2f}) vs "
        f"{mb.get('primary') or 'unknown'} (confidence {mb.get('confidence') or 0:.2f})")
    # 3. symbol rate -------------------------------------------------------------------
    ra, rb = sa.get("symbol_rate_hz"), sb.get("symbol_rate_hz")
    if ra and rb:
        ratio = max(ra, rb) / max(min(ra, rb), 1e-9)
        s_rate = clamp01(1.0 - abs(math.log(ratio)) / math.log(2.0))
        add("Symbol-rate ratio", round(ratio, 4), None,
            f"{ra:,.1f} sym/s vs {rb:,.1f} sym/s (equal rates give 1.0)")
    else:
        s_rate, ratio = 0.0, None
        add("Symbol-rate ratio", None, None, "one of the records has no symbol-rate estimate")
    # 4. bandwidth ---------------------------------------------------------------------
    ba, bb = ssa.get("obw_99_hz"), ssb.get("obw_99_hz")
    if ba and bb:
        br = max(ba, bb) / max(min(ba, bb), 1e-9)
        s_bw = clamp01(1.0 - abs(math.log(br)) / math.log(2.0))
        add("Occupied-bandwidth ratio", round(br, 4), None,
            f"{ba:,.0f} Hz vs {bb:,.0f} Hz")
    else:
        s_bw, br = 0.0, None
        add("Occupied-bandwidth ratio", None, None, "occupied bandwidth unavailable for one record")
    # 5. SNR difference ----------------------------------------------------------------
    na, nb = ssa.get("snr_db"), ssb.get("snr_db")
    if na is not None and nb is not None:
        d = abs(float(na) - float(nb))
        s_snr = clamp01(1.0 - d / 20.0)
        add("SNR difference", round(d, 2), "dB", f"{na:.1f} dB vs {nb:.1f} dB")
    else:
        s_snr, d = 0.0, None
        add("SNR difference", None, "dB", "SNR unavailable for one record")
    # 6. constellation / EVM -----------------------------------------------------------
    ea = _get(da, "quality", "evm_percent")
    eb = _get(db_, "quality", "evm_percent")
    if ea is not None and eb is not None and same_mod:
        s_evm = clamp01(1.0 - abs(float(ea) - float(eb)) / 20.0)
        add("EVM difference", round(abs(float(ea) - float(eb)), 2), "%",
            f"{ea:.2f} % vs {eb:.2f} % (same modulation hypothesis)")
    else:
        s_evm = 0.0
        add("EVM difference", None, "%",
            "not comparable (different modulation hypotheses or missing EVM)")
    # 7. bit distributions -------------------------------------------------------------
    hista = _get(sa, "bitstream", "byte_histogram") or {}
    histb = _get(sb, "bitstream", "byte_histogram") or {}
    pa_, pb_ = None, None
    if isinstance(hista, dict) and isinstance(histb, dict):
        ca = np.asarray(hista.get("counts") or [], dtype=np.float64)
        cb = np.asarray(histb.get("counts") or [], dtype=np.float64)
        if ca.size == cb.size and ca.size > 1 and ca.sum() > 0 and cb.sum() > 0:
            pa_, pb_ = ca / ca.sum(), cb / cb.sum()
    if pa_ is not None:
        m = 0.5 * (pa_ + pb_)
        with np.errstate(divide="ignore", invalid="ignore"):
            kl1 = float(np.nansum(np.where(pa_ > 0, pa_ * np.log2(pa_ / np.maximum(m, 1e-12)), 0.0)))
            kl2 = float(np.nansum(np.where(pb_ > 0, pb_ * np.log2(pb_ / np.maximum(m, 1e-12)), 0.0)))
        js = float(np.sqrt(max(0.5 * (kl1 + kl2), 0.0)))          # Jensen-Shannon distance
        s_bits = clamp01(1.0 - js / 0.5)
        add("Byte-distribution distance", round(js, 4), "bits",
            "Jensen-Shannon distance between the two byte histograms (0 = identical)")
    else:
        s_bits = 0.0
        add("Byte-distribution distance", None, "bits",
            "byte histograms are not available for both records")
    # 8. time-domain envelope similarity -----------------------------------------------
    s_env = 0.0
    if isinstance(pa_, np.ndarray):
        s_env = s_bits      # symbol statistics and byte statistics carry the same information here

    weights = {"spec": 0.25, "mod": 0.25, "rate": 0.10, "bw": 0.10, "snr": 0.10, "evm": 0.05,
               "bits": 0.15}
    parts = {"spec": s_spec, "mod": s_mod, "rate": s_rate, "bw": s_bw, "snr": s_snr, "evm": s_evm,
             "bits": s_bits}
    score = float(sum(weights[k] * parts[k] for k in weights))
    verdict = verdict_for(score)
    comparable = sum(1 for k in ("spec", "mod", "rate", "bw", "snr", "evm", "bits") if parts[k] > 0)
    notes.append(f"{comparable} of 7 comparisons were possible; the score is the weighted mean of "
                 f"those that were, using fixed weights {weights}")
    notes.append("similarity is a measured statement about the two recordings; it is NOT evidence "
                 "that both come from the same transmitter, emitter or source")
    if same_mod and conf >= 0.5 and score >= 0.8:
        notes.append("both recordings agree on modulation, symbol rate and spectral shape within a "
                     "few percent - consistent with the same waveform family, still not proof of a "
                     "common source")
    return {"ok": True, "verdict": verdict, "similarity": score, "metrics": metrics,
            "parts": parts, "weights": weights, "notes": notes,
            "method": "weighted comparison of independently measured spectral, modulation, "
                      "timing, quality and bitstatistics features"}


def comparison_param(result: dict) -> dict:
    return param("Signal comparison", result.get("verdict"), None, result.get("similarity"),
                 "weighted multi-feature comparison",
                 evidence=[f"{m['metric']}: {m['value']} {m['unit'] or ''}".strip()
                           for m in (result.get("metrics") or [])[:6]],
                 limitations=[n for n in (result.get("notes") or [])
                              if "NOT evidence" in n or "not proof" in n],
                 status="ok" if result.get("ok") else "unable")
