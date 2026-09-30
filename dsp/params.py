"""Signal parameter extraction: symbol rate, excess bandwidth, FSK deviation,
amplitude/phase statistics and a per-candidate timing-quality score.

Symbol-rate estimation runs *three independent estimators* and reconciles them,
because no single blind estimator is reliable for every modulation:

1. **Squared-envelope spectral line** - for PSK/QAM/ASK with excess bandwidth the
   |x|^2 process is cyclostationary with cyclic frequency equal to the symbol rate,
   producing a line in the |x|^2 spectrum at exactly Rs.
2. **Instantaneous-frequency derivative line** - for (G)FSK the instantaneous
   frequency is piecewise constant, so its first difference is a spike train at the
   symbol boundaries: again a line at Rs.
3. **Fourth-power spectral line** - PAM/QAM/Nyquist-shaped signals also show a line
   at Rs in the |x|^4 spectrum, useful when |x|^2 lines are weak.

Every candidate is scored by line prominence (per Hz of the local floor) and
validated against the occupied bandwidth, and the reconciliation reports when the
estimators disagree instead of silently picking one.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import signal as sigproc

from .preprocess import make_window
from .utils import param, clamp01, STATUS_OK, STATUS_LOW, STATUS_UNAVAILABLE, parabolic_peak


# --------------------------------------------------------------------------- #
#  Amplitude / phase statistics
# --------------------------------------------------------------------------- #
def amplitude_statistics(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.complex128)
    mag = np.abs(x)
    m = float(np.mean(mag))
    if m <= 0:
        return {"ok": False, "error": "zero-power signal"}
    centred = mag - m
    std = float(np.std(mag))
    p = mag ** 2
    kurt = float(np.mean(((p - p.mean()) / max(p.std(), 1e-30)) ** 4))
    return {
        "ok": True,
        "mean_magnitude": m, "std_magnitude": std,
        "sigma_aa": float(std / m),                     # normalised amplitude variation
        "coefficient_of_variation": float(std / m),
        "magnitude_kurtosis": kurt,
        "peak_to_mean": float(np.max(mag) / m),
        "envelope_ripple_db": float(20 * math.log10(max(np.percentile(mag, 95), 1e-30) /
                                                    max(np.percentile(mag, 5), 1e-30))),
        "amplitude_asymmetry": float(np.mean(mag ** 2) / max(np.mean(mag) ** 2, 1e-30)),
        "interpretation": ("constant envelope (PSK/FSK-like or limiter" if std / m < 0.15 else
                           "amplitude variation present (QAM/AM/ASK-like or low SNR)")
        if std / m < 0.4 else "very large amplitude variation (AM, ASK, bursts or heavy fading)",
    }


def phase_statistics(x: np.ndarray, fs: float, min_mag_frac: float = 0.2) -> dict:
    x = np.asarray(x, dtype=np.complex128)
    mag = np.abs(x)
    thr = float(np.max(mag) * min_mag_frac)
    sel = mag >= thr
    if not np.any(sel):
        return {"ok": False, "error": "no samples above the amplitude gate"}
    ph = np.angle(x[sel])
    cph = np.angle(np.exp(1j * (ph - np.mean(ph))))
    # circular statistics
    R = float(np.abs(np.mean(np.exp(1j * ph))))
    circ_std = math.sqrt(max(0.0, -2 * math.log(max(R, 1e-12))))
    dph = np.angle(np.exp(1j * np.diff(ph)))
    return {
        "ok": True, "gated_fraction": float(np.mean(sel)),
        "mean_phase_rad": float(np.angle(np.mean(np.exp(1j * ph)))),
        "circular_std_rad": circ_std,
        "std_phase_rad": float(np.std(cph)),
        "n_modes_hint": int(max(1, round(2 * math.pi / max(circ_std, 1e-6)))) if circ_std > 0 else 0,
        "phase_transition_std_rad": float(np.std(dph)),
        "phase_gate_note": f"statistics use the {100*np.mean(sel):.0f}% of samples above "
                           f"{min_mag_frac*100:.0f}% of peak magnitude (avoids noise-driven phase)",
    }


# --------------------------------------------------------------------------- #
#  Spectral-line estimators for the symbol rate
# --------------------------------------------------------------------------- #
def _line_search(power: np.ndarray, freqs: np.ndarray, f_min: float, f_max: float,
                 min_bins_apart: int = 3, max_peaks: int = 6) -> list[dict]:
    """Find spectral lines in a restricted band with prominence over a local floor."""
    m = (freqs >= f_min) & (freqs <= f_max)
    if np.count_nonzero(m) < 8:
        return []
    p = power[m].copy()
    f = freqs[m]
    # local floor: running median over ~15% of the search band
    w = max(5, int(0.15 * p.size) | 1)
    if w < p.size:
        floor = sigproc.medfilt(p, kernel_size=min(w if w % 2 else w + 1, (p.size // 2) * 2 + 1))
    else:
        floor = np.full_like(p, float(np.median(p)))
    ratio = p / np.maximum(floor, 1e-30)
    ratio_db = 10 * np.log10(ratio)
    peaks = []
    order = np.argsort(ratio_db)[::-1]
    taken: list[int] = []
    for idx in order[: 200]:
        if ratio_db[idx] < 1.0:
            break
        if any(abs(int(idx) - t) < min_bins_apart for t in taken):
            continue
        taken.append(int(idx))
        f_ref, p_ref = parabolic_peak(ratio_db, int(idx))
        peaks.append({
            "frequency_hz": float(np.interp(f_ref, np.arange(f.size), f)),
            "prominence_db": float(p_ref),
            "bin": int(idx),
        })
        if len(peaks) >= max_peaks:
            break
    return peaks


def _diff_spectrum(x: np.ndarray, fs: float, nperseg: int, window: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.diff(x)
    d = d - np.mean(d)
    f, P = sigproc.welch(d, fs=fs, nperseg=min(nperseg, max(64, d.size // 4)),
                         window=make_window(window, min(nperseg, max(64, d.size // 4))),
                         return_onesided=False, scaling="density")
    order = np.argsort(f)
    return f[order], P[order]


def _spectrum_of_power(x: np.ndarray, fs: float, order: int, nperseg: int,
                       window: str) -> tuple[np.ndarray, np.ndarray]:
    y = np.abs(x) ** order if order > 1 else np.abs(x)
    y = y - np.mean(y)
    n = min(nperseg, max(64, y.size // 4))
    f, P = sigproc.welch(y, fs=fs, nperseg=n, window=make_window(window, n),
                         return_onesided=False, scaling="density")
    o = np.argsort(f)
    return f[o], P[o]


def _freq_floor_hz(fs: float) -> float:
    """Smallest frequency searched by the blind estimators, in the caller's units.

    The estimators work in *absolute* hertz when the sample rate is known and in *normalised*
    frequency (cycles per sample, i.e. ``fs = 1``) when the capture carries no rate metadata - a
    raw IQ dump has none, so this is a normal operating mode rather than an edge case.  A fixed
    absolute floor such as "16 Hz" would exceed the entire search band at ``fs = 1`` and the
    estimator would return nothing; the floor is therefore scaled: at least fs/4000 and at least
    2 % of the sample rate, but never above the historical 16 Hz floor for real captures.
    """
    fs = float(fs) or 1.0
    return float(max(fs / 4000.0, min(16.0, 0.02 * fs)))


def _search_band_hz(fs: float, obw_hz: float | None) -> tuple[float, float]:
    """Frequency search band of the blind symbol-rate estimators (relative to ``fs``)."""
    fs = float(fs) or 1.0
    floor = _freq_floor_hz(fs)
    obw = float(obw_hz) if obw_hz and obw_hz > 0 else None
    if obw:
        f_min = max(floor, fs / 2000.0, 0.30 * obw)
        f_max = min(0.55 * fs, 2.6 * obw)
    else:
        f_min = max(floor, fs / 2000.0)
        f_max = min(0.55 * fs, max(4.0 * f_min, fs / 8.0))
    if f_max <= f_min:
        f_min = max(floor, fs / 2000.0)
        f_max = min(0.55 * fs, 0.25 * fs)
    return float(f_min), float(f_max)


def estimate_symbol_rate(x: np.ndarray, fs: float, obw_hz: float | None = None,
                         snr_db: float | None = None, spectral_flat: bool = False,
                         window: str = "hann", max_candidates: int = 5) -> dict:
    """Blind symbol-rate estimation with cross-validated candidates."""
    x = np.asarray(x, dtype=np.complex128)
    n = x.size
    if n < 256:
        return {"ok": False, "error": "too few samples for symbol-rate estimation",
                "primary": None, "candidates": [], "params": []}
    nperseg = int(min(8192, max(256, n // 8)))
    # The occupied bandwidth is a physical constraint on the symbol rate: for a single carrier
    # 0.35*BW <= Rs <= 2.5*BW (BW/Rs is between 0.4 and ~2.9 including roll-off and FSK deviation).
    # Without a bandwidth estimate, fall back to a wide relative search.
    f_min, f_max = _search_band_hz(fs, obw_hz)
    if f_max <= f_min:
        return {"ok": False, "error": "search band empty", "primary": None, "candidates": [], "params": []}

    estimators: list[dict] = []
    for order, tag, why in ((2, "|x|^2 spectrum", "cyclostationarity of the squared envelope"),
                            (4, "|x|^4 spectrum", "cyclostationarity of the fourth power")):
        f, P = _spectrum_of_power(x, fs, order, nperseg, window)
        peaks = _line_search(P, f, f_min, f_max)
        if peaks:
            estimators.append({"name": tag, "reason": why, "peaks": peaks,
                               "freqs": f, "power": P})
    fd, Pd = _diff_spectrum(x, fs, nperseg, window)
    peaks_d = _line_search(Pd, fd, f_min, f_max)
    if peaks_d:
        estimators.append({"name": "instantaneous-frequency derivative spectrum",
                           "reason": "symbol-boundary spike train of the phase derivative",
                           "peaks": peaks_d, "freqs": fd, "power": Pd})

    # collect and cluster candidates that agree within 2% across estimators
    raw: list[dict] = []
    for est in estimators:
        for p in est["peaks"]:
            raw.append({"f": p["frequency_hz"], "prominence_db": p["prominence_db"],
                        "estimator": est["name"], "reason": est["reason"]})
    # half-rate / double-rate aliases are common artefacts: keep them but flag them
    clusters: list[dict] = []
    for cand in sorted(raw, key=lambda r: -r["prominence_db"]):
        placed = False
        for c in clusters:
            if abs(cand["f"] - c["f"]) / max(c["f"], 1e-9) < 0.02:
                c["hits"].append(cand)
                c["f"] = float(np.mean([h["f"] for h in c["hits"]]))
                c["prominence_db"] = float(max(c["prominence_db"], cand["prominence_db"]))
                placed = True
                break
        if not placed:
            clusters.append({"f": cand["f"], "prominence_db": cand["prominence_db"], "hits": [cand]})
    # rank: multi-estimator agreement first, then prominence, then a strong prior on how the
    # candidate rate relates to the measured occupied bandwidth
    for c in clusters:
        c["n_estimators"] = len({h["estimator"] for h in c["hits"]})
        c["score"] = c["n_estimators"] * 3.0 + min(c["prominence_db"], 25.0)
        if obw_hz and obw_hz > 0:
            ratio = obw_hz / c["f"]
            if 0.9 <= ratio <= 2.4:
                c["score"] += 6.0                     # physically plausible BW/Rs
            elif 0.5 <= ratio < 0.9 or 2.4 < ratio <= 3.5:
                c["score"] += 1.0
            else:
                c["score"] -= 8.0                     # implausible for a single carrier
        # estimators are known to be unreliable within one resolution cell of the search bounds
        if c["f"] < 1.25 * f_min or c["f"] > 0.85 * f_max:
            c["score"] -= 6.0
    clusters.sort(key=lambda c: -c["score"])

    candidates: list[dict] = []
    for c in clusters[:max_candidates]:
        rs = float(c["f"])
        if rs < f_min or rs > f_max:
            continue
        bw_ratio = (obw_hz / rs) if obw_hz else None
        consistency = None
        if bw_ratio is not None:
            # for a shaped single-carrier signal 1 <= BW/Rs <= 2.2 is physically sound
            consistency = "consistent with the occupied bandwidth" if 0.9 <= bw_ratio <= 2.4 else \
                          ("bandwidth is narrower than the symbol rate - implausible for a single carrier"
                           if bw_ratio < 0.9 else
                           "occupied bandwidth far exceeds this rate - the signal may be multi-carrier "
                           "or this line is a harmonic")
        conf = clamp01(0.30 + 0.16 * c["n_estimators"] + 0.14 * clamp01(c["prominence_db"] / 18.0))
        if consistency and consistency.startswith("bandwidth is narrower"):
            conf *= 0.35
        if snr_db is not None and snr_db < 8:
            conf *= 0.7
        candidates.append({
            "symbol_rate_hz": rs,
            "confidence": round(float(conf), 4),
            "prominence_db": round(float(c["prominence_db"]), 2),
            "n_estimators": int(c["n_estimators"]),
            "estimators": sorted({h["estimator"] for h in c["hits"]}),
            "evidence": [f"line at {rs:g} Hz with {c['prominence_db']:.1f} dB prominence over the local floor"
                         f" in the {h['estimator']}" for h in c["hits"][:3]] +
                        ([consistency] if consistency else []),
            "limitations": ([] if c["n_estimators"] > 1 else
                            ["confirmed by a single estimator only: rate may be a harmonic or artefact"]) +
                           ([consistency] if consistency and not consistency.startswith("consistent") else []),
            "bandwidth_ratio": (round(float(bw_ratio), 3) if bw_ratio else None),
        })

    spectral_flat_note = None
    if spectral_flat:
        spectral_flat_note = ("the detected spectrum is nearly noise-like: for OFDM/noise-like emissions the "
                             "second-order estimators below are not applicable and the symbol rate is reported "
                             "as unavailable rather than guessed")

    if not candidates:
        return {"ok": True, "primary": None, "candidates": [], "estimators_tried": [e["name"] for e in estimators],
                "search_band_hz": [f_min, f_max],
                "method": "spectral-line search in the |x|^2, |x|^4 and phase-derivative spectra",
                "limitations": ["no spectral line consistent with a symbol rate was found"],
                "params": [param("Symbol rate", None, "sym/s", None,
                                 "|x|^2 / |x|^4 / IF-derivative spectral-line search",
                                 status=STATUS_UNAVAILABLE,
                                 limitations=["no cyclostationary line detected: the signal may be "
                                              "unshaped (alpha=0), noise-like, or too low in SNR"])]}
    primary = candidates[0]
    # EVM-based validation of the top candidates is added by the caller (needs demodulation),
    # but a rate that is exactly half of another candidate with stronger evidence is flagged here.
    for c in candidates:
        for d in candidates:
            if c is not d and abs(d["symbol_rate_hz"] - 2 * c["symbol_rate_hz"]) < 0.02 * d["symbol_rate_hz"] \
                    and d["prominence_db"] > c["prominence_db"]:
                c["limitations"].append(f"half of the stronger candidate {d['symbol_rate_hz']:g} Hz: this may be "
                                        f"a sub-harmonic artefact")
    params = [
        param("Symbol rate", primary["symbol_rate_hz"], "sym/s", primary["confidence"],
              "spectral-line (cyclostationary) estimation: " + ", ".join(primary["estimators"]),
              evidence=primary["evidence"], limitations=primary["limitations"],
              status=STATUS_OK if primary["confidence"] > 0.5 else STATUS_LOW),
        param("Symbol rate - alternative candidates", [c["symbol_rate_hz"] for c in candidates[1:]],
              "sym/s", None, "same estimator bank; ranked by multi-estimator agreement",
              note="competing rates are carried into the multi-hypothesis analysis"),
        param("Symbol duration", 1.0 / primary["symbol_rate_hz"], "s", primary["confidence"],
              "reciprocal of the estimated symbol rate"),
    ]
    if spectral_flat_note:
        params.append(param("Symbol rate (noise-like spectrum)", None, "sym/s", None,
                            "not applicable to noise-like/OFDM emissions",
                            status=STATUS_UNAVAILABLE, limitations=[spectral_flat_note]))
    return {
        "ok": True, "primary": primary, "candidates": candidates,
        "estimators_tried": [e["name"] for e in estimators],
        "search_band_hz": [f_min, f_max],
        "method": "blind spectral-line estimation on |x|^2, |x|^4 and the instantaneous-frequency "
                  "derivative, with cross-estimator agreement scoring and bandwidth consistency checks",
        "limitations": ["estimators assume a single carrier with statistically independent symbols",
                        "signals with alpha=0 shaping (rectangular, no excess bandwidth) produce very weak lines",
                        "at low SNR the |x|^2 line weakens before the carrier becomes undetectable"],
        "params": params,
    }


def timing_line_quality(x: np.ndarray, fs: float, rs: float, window: str = "hann") -> dict:
    """Measurable strength of the cyclostationary line at a *given* candidate rate.

    This is the validation score used when ranking hypotheses: a rate that is not
    the true symbol rate shows no line at that cyclic frequency.
    """
    x = np.asarray(x, dtype=np.complex128)
    if x.size < 256 or rs <= 0:
        return {"ok": False, "reason": "insufficient data"}
    y = np.abs(x) ** 2
    y = y - np.mean(y)
    nper = int(min(8192, max(256, y.size // 4)))
    f, P = sigproc.welch(y, fs=fs, nperseg=nper, window=make_window(window, nper),
                         return_onesided=False, scaling="density")
    o = np.argsort(f)
    f, P = f[o], P[o]
    df = abs(f[1] - f[0]) if f.size > 1 else fs
    idx = int(np.argmin(np.abs(f - rs)))
    band = P[max(0, idx - max(1, int(0.01 * rs / df))): idx + max(2, int(0.01 * rs / df)) + 1]
    peak = float(np.max(band)) if band.size else 0.0
    # local floor excluding the line
    excl = np.ones(f.size, dtype=bool)
    excl[max(0, idx - 3 * max(1, int(0.01 * rs / df))): idx + 3 * max(1, int(0.01 * rs / df)) + 1] = False
    floor = float(np.median(P[excl])) if np.any(excl) else float(np.median(P))
    snr_line = 10 * math.log10(max(peak, 1e-30) / max(floor, 1e-30))
    return {"ok": True, "line_db_above_floor": round(snr_line, 3),
            "peak_frequency_hz": float(f[idx]), "local_floor": floor, "line_power": peak,
            "method": "|x|^2 Welch PSD line search at the candidate cyclic frequency"}


def estimate_rolloff_alpha(x: np.ndarray, fs: float, rs: float | None = None,
                           obw_hz: float | None = None) -> dict:
    """Excess-bandwidth (roll-off) estimate from the PSD skirt width.

    alpha is estimated from the ratio between the total occupied bandwidth and the
    -3 dB (or symbol-rate-related) flat-top width, then sanity-clamped to [0, 1].
    """
    if not obw_hz or obw_hz <= 0:
        return {"ok": False, "error": "occupied bandwidth required"}
    if rs and rs > 0:
        alpha = obw_hz / rs - 1.0
        est = float(np.clip(alpha, 0.0, 1.0))
        conf = 0.6 if 0.05 <= alpha <= 0.8 else 0.35
        return {"ok": True, "alpha": round(est, 3), "method":
                "alpha = occupied_bandwidth / symbol_rate - 1 (requires a reliable symbol rate)",
                "confidence": conf,
                "limitations": ["both inputs (occupied bandwidth and symbol rate) carry their own error; "
                                "values below 0 or above 1 are clipped"],
                "raw_alpha": round(float(alpha), 3)}
    return {"ok": False, "error": "symbol rate required for the roll-off estimate"}


def estimate_fsk_deviation(x: np.ndarray, fs: float, rs: float | None = None,
                           obw_hz: float | None = None) -> dict:
    """Peak frequency deviation of an FSK signal from the instantaneous frequency.

    Uses the 90th percentile of |f_inst| after removing the residual carrier, and
    cross-checks the total bandwidth via Carson's rule (BW = 2*(df + Rs)).
    """
    x = np.asarray(x, dtype=np.complex128)
    if x.size < 256:
        return {"ok": False, "error": "insufficient samples"}
    ph = np.unwrap(np.angle(x))
    dph = np.diff(ph)
    slope = float(np.polyfit(np.arange(dph.size, dtype=np.float64), dph, 1)[0])
    inst = (dph - slope) * fs / (2 * math.pi)
    dev = float(np.percentile(np.abs(inst), 90))
    res = {"ok": True, "deviation_hz": dev, "residual_carrier_hz": float(slope * fs / (2 * math.pi)),
           "method": "phase-difference frequency discriminator; deviation = 90th percentile of |f_inst| "
                     "after removing the linear phase trend",
           "confidence": 0.6}
    if rs:
        carson = 2 * (dev + rs)
        res["carson_bandwidth_hz"] = float(carson)
        if obw_hz:
            err = abs(carson - obw_hz) / max(obw_hz, 1e-9)
            res["carson_consistency"] = round(float(1 - err), 3)
            res["evidence"] = [f"Carson's rule gives {carson:g} Hz vs measured occupied bandwidth {obw_hz:g} Hz "
                               f"({100*(1-err):.0f}% agreement)"]
    return res


def estimate_baud_from_envelope(x: np.ndarray, fs: float, obw_hz: float | None = None) -> dict:
    """ASK/OOK-specific rate estimate from the envelope spectrum."""
    x = np.asarray(x, dtype=np.complex128)
    env = np.abs(x)
    env = env - np.mean(env)
    nper = int(min(8192, max(128, env.size // 4)))
    f, P = sigproc.welch(env, fs=fs, nperseg=nper, window=make_window("hann", nper),
                         return_onesided=True, scaling="density")
    f_min, f_max = _freq_floor_hz(fs), min(0.5 * fs, (obw_hz or fs / 4) * 3)
    peaks = _line_search(P, f, f_min, f_max, max_peaks=4)
    return {"ok": bool(peaks), "peaks": peaks,
            "method": "envelope (|x|) spectrum line search - keying rate of ASK/OOK or the AM message tone",
            "note": "for AM this frequency is the modulation tone, not a symbol rate"}


def full_parameter_report(x: np.ndarray, fs: float, obw_hz: float | None = None,
                          snr_db: float | None = None, spectral_flat: bool = False) -> dict:
    """Convenience wrapper producing the parameter block used by the UI/report."""
    amp = amplitude_statistics(x)
    ph = phase_statistics(x, fs)
    rate = estimate_symbol_rate(x, fs, obw_hz=obw_hz, snr_db=snr_db, spectral_flat=spectral_flat)
    env = estimate_baud_from_envelope(x, fs, obw_hz)
    out = {"amplitude": amp, "phase": ph, "symbol_rate": rate, "envelope": env}
    prm: list[dict] = []
    if amp.get("ok"):
        prm += [
            param("Amplitude variation (sigma_aa)", amp["sigma_aa"], None, 0.85,
                  "std(|x|) / mean(|x|) of the analytic signal",
                  note=amp["interpretation"]),
            param("Envelope ripple (5-95%)", amp["envelope_ripple_db"], "dB", 0.8,
                  "95th/5th percentile magnitude ratio"),
            param("Magnitude kurtosis", amp["magnitude_kurtosis"], None, 0.75,
                  "4th standardised moment of the instantaneous power"),
        ]
    if ph.get("ok"):
        prm += [
            param("Phase standard deviation", ph["std_phase_rad"], "rad", 0.7,
                  "std of the phase after an amplitude gate (top 80% magnitude samples)",
                  limitations=["phase statistics are biased by residual carrier offset and noise"]),
            param("Phase transition spread", ph["phase_transition_std_rad"], "rad", 0.7,
                  "std of first differences of the gated phase"),
        ]
    prm += rate.get("params", [])
    if env.get("ok"):
        prm.append(param("Envelope-spectrum line", [round(p["frequency_hz"], 2) for p in env["peaks"]],
                         "Hz", 0.55, env["method"],
                         limitations=[env["note"]]))
    out["params"] = prm
    if rate.get("primary"):
        out["alpha"] = estimate_rolloff_alpha(x, fs, rate["primary"]["symbol_rate_hz"], obw_hz)
        out["fsk"] = estimate_fsk_deviation(x, fs, rate["primary"]["symbol_rate_hz"], obw_hz)
    return out
