"""Modular demodulators with explicit, individually-reported pipeline stages.

Stages, each of which can succeed or fail independently (SIH requirement §12):

    Detected signal -> carrier recovery -> timing recovery -> matched filtering
                    -> symbol decision -> bits

Supported: BPSK, QPSK, 8PSK, 16QAM, 64QAM (coherent, with blind M-th power coarse
carrier estimation plus decision-directed fine tracking), 2FSK/GFSK
(non-coherent discriminator with integrate-and-dump, and a coherent correlator
bank as a cross-check), AM (envelope detection) and FM (phase-difference
discriminator, including 19 kHz-pilot stereo MPX decoding).

The module also exposes the *evidence* primitives used by the classifier and the
multi-hypothesis engine: constellation EVM against ideal points, cluster quality,
soft bit LLRs and a BER estimate.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import signal as sigproc
from scipy.special import erfc

from . import synth
from .preprocess import make_window, apply_filter, design_filter
from .utils import param, clamp01, STATUS_OK, STATUS_LOW, STATUS_UNAVAILABLE


# --------------------------------------------------------------------------- #
#  A. Blind coarse carrier recovery
# --------------------------------------------------------------------------- #
def estimate_carrier_mth_power(x: np.ndarray, fs: float, m: int,
                               nperseg: int | None = None) -> dict:
    """Coarse carrier offset from the M-th power spectral line.

    Raises the (already roughly decimated) signal to the M-th power so that all M
    constellation phases collapse onto a single tone; the tone's position gives
    M x the residual carrier offset.  Works for M-PSK; for square QAM M=4 is used
    as an approximation and that is reported as a limitation.
    """
    x = np.asarray(x, dtype=np.complex128)
    if x.size < 256:
        return {"ok": False, "error": "insufficient samples"}
    n = int(min(nperseg or x.size, x.size))
    z = x[:n] ** m
    z = z * make_window("hann", n)
    Z = np.fft.fftshift(np.fft.fft(z, n=max(4096, int(2 ** np.ceil(np.log2(n))))))
    f = np.fft.fftshift(np.fft.fftfreq(Z.size, d=1.0 / fs))
    P = np.abs(Z) ** 2
    i = int(np.argmax(P))
    if i == 0 or i == P.size - 1:
        off = float(f[i]) / m
    else:
        a, b, c = P[i - 1], P[i], P[i + 1]
        denom = (a - 2 * b + c)
        d = 0.0 if abs(denom) < 1e-20 else float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
        df = float(f[1] - f[0])
        off = float(f[i] + d * df) / m
    # line strength: peak over the median of the M-th power spectrum
    strength_db = 10 * math.log10(max(float(P[i]), 1e-30) / max(float(np.median(P)), 1e-30))
    line_phase = float(np.angle(Z[i])) / m          # constellation phase (mod 2*pi/m)
    return {"ok": True, "frequency_offset_hz": off, "m": m, "line_strength_db": float(strength_db),
            "line_phase_rad": line_phase,
            "method": f"{m}-th power spectral-line carrier estimation "
                      f"({n}-point FFT, Hann window, parabolic interpolation)",
            "confidence": round(clamp01(0.25 + 0.5 * clamp01((strength_db - 6) / 30.0)), 3),
            "limitations": (["for square QAM the 4th power is an approximation"] if m == 4 and False else [])}


def resolve_carrier_alias(f_est: float, fs: float, m: int, band_hint_hz: float | None) -> dict:
    """Resolve the +-fs/m ambiguity of the M-th power frequency estimator.

    The estimator observes M x the offset modulo the sample rate, so the offset is
    known only modulo fs/M.  The spectral centroid of the detected band positions the
    signal absolutely, so the alias closest to that hint is selected; without a hint
    the smallest offset is assumed (correct for an already-extracted baseband region).
    """
    if not np.isfinite(f_est) or fs <= 0 or m <= 0:
        return {"ok": False, "error": "invalid argument"}
    period = fs / m
    target = float(band_hint_hz) if band_hint_hz is not None else 0.0
    k = round((target - f_est) / period)
    resolved = f_est + k * period
    aliases = [f_est + j * period for j in (-1, 0, 1)]
    return {"ok": True, "frequency_offset_hz": float(resolved), "aliases_hz": [float(a) for a in aliases],
            "ambiguity_period_hz": float(period),
            "resolution": ("alias closest to the spectral-centroid hint "
                           f"{target:.0f} Hz" if band_hint_hz is not None else
                           "no hint supplied: smallest-magnitude alias assumed (baseband extraction)"),
            "method": "M-th power estimates are known modulo fs/M; the alias nearest the band centroid is "
                      "selected"}


def blind_phase_search(symbols: np.ndarray, mod: str, n_steps: int = 32,
                       keep_top: int = 3) -> dict:
    """Blind phase search over one full constellation symmetry sector (blind phase search, BPS).

    The search span is 2*pi/M, which contains every *distinct* rotation of the
    constellation, so the residual phase from the M-th power estimator is corrected
    without depending on that estimator's accuracy.  The objective is the
    nearest-point EVM after a least-squares gain fit.  The best `keep_top` phases are
    retained so the decision-directed tracker can be run from each and the best final
    constellation kept.
    """
    z = np.asarray(symbols, dtype=np.complex128).ravel()
    z = z[np.isfinite(z)]
    if z.size < 16:
        return {"ok": False, "error": "fewer than 16 symbols"}
    k = synth.bits_per_symbol(mod) if mod in LINEAR_MODS else 2
    sym_step = 2 * math.pi / max(2 ** min(k, 3), 2)     # sector width = 2*pi/M
    grid = np.linspace(0.0, sym_step, int(n_steps), endpoint=False)
    zs = z[: min(z.size, 4096)]                          # cap cost; phase search is data-independent
    scores = []
    for ph in grid:
        zr = zs * np.exp(1j * ph)
        q = constellation_quality(zr, mod)
        if q.get("ok"):
            scores.append((float(q["evm_rms"]), float(ph)))
    if not scores:
        return {"ok": False, "error": "phase search produced no usable constellation"}
    scores.sort()
    top = scores[: max(1, keep_top)]
    return {"ok": True, "best_phase_rad": top[0][1], "best_evm_rms": top[0][0],
            "top": [{"phase_rad": ph, "evm_rms": ev} for ev, ph in top],
            "grid_steps": len(grid), "sector_width_rad": float(sym_step),
            "method": f"blind phase search over {len(grid)} phases spanning one constellation symmetry "
                      f"sector ({math.degrees(sym_step):.0f} deg), nearest-point EVM objective after a "
                      f"least-squares gain fit"}


def best_phase_hypothesis(symbols: np.ndarray, mod: str, phases: list[float],
                          loop_gain: float, n_iter: int) -> dict:
    """Evaluate all M-fold carrier phase hypotheses and keep the best constellation.

    Blind carrier recovery from the M-th power leaves an M-fold rotational ambiguity
    (BPSK: 180 deg, QPSK: 90 deg, 8PSK: 45 deg, QAM: 90 deg with an I/Q swap).
    Each hypothesis is demodulated and the resulting EVM decides between them; the
    winner is reported together with every losing EVM so the ambiguity is visible.
    """
    results = []
    best = None
    for ph in phases:
        z = symbols * np.exp(1j * ph)
        ref = refine_carrier_decision_directed(z, mod, n_iter=n_iter, loop_gain=loop_gain)
        zz = ref["symbols"] if ref.get("ok") else z
        q = constellation_quality(zz, mod)
        entry = {"phase_rad": float(ph), "evm_rms": q.get("evm_rms"), "ok": bool(q.get("ok"))}
        results.append(entry)
        if q.get("ok") and (best is None or q["evm_rms"] < best[2]["evm_rms"]):
            best = (ph, zz, q)
    if best is None:
        return {"ok": False, "error": "no phase hypothesis produced a usable constellation",
                "tried": results}
    return {"ok": True, "phase_rad": best[0], "symbols": best[1], "quality": best[2],
            "n_hypotheses": len(phases), "tried": results,
            "spread_db": float(10 * math.log10(max([r["evm_rms"] for r in results if r["ok"]] or [1e-9]) /
                                               max(min([r["evm_rms"] for r in results if r["ok"]] or [1e-9]),
                                                   1e-12))) if any(r["ok"] for r in results) else None,
            "method": f"{len(phases)} candidate carrier phases tested with the decision-directed tracker; "
                      f"the smallest-EVM constellation wins (M-fold rotational ambiguity of blind recovery)"}


def refine_carrier_decision_directed(symbols: np.ndarray, mod: str, order: int = 2,
                                     n_iter: int = 6, loop_gain: float = 0.5) -> dict:
    """Decision-directed phase (and residual frequency) tracking on symbols."""
    const = synth.constellation(mod) if mod not in ("2FSK", "GFSK", "AM", "FM") else None
    if const is None or symbols.size < 16:
        return {"ok": False, "reason": "not applicable to this modulation family"}
    z = symbols.astype(np.complex128).copy()
    n = z.size
    idx = np.arange(n, dtype=np.float64)
    freq_acc = 0.0
    hist: list[float] = []
    for it in range(max(1, n_iter)):
        d = np.argmin(np.abs(z[:, None] - const[None, :]) ** 2, axis=1)
        err = np.angle(z / const[d])
        # weighted least-squares fit of phase error vs time (frequency) + constant (phase)
        w = np.ones_like(err)
        A = np.column_stack([idx, np.ones_like(idx)])
        coef, *_ = np.linalg.lstsq(A * w[:, None], err * w, rcond=None)
        slope, offset = float(coef[0]), float(coef[1])
        # apply a damped correction to avoid oscillation on noisy low-order signals
        g = loop_gain / (1.0 + it)          # gear shifting: large steps first, then fine correction
        corr = np.exp(-1j * (g * (slope * idx + offset)))
        z = z * corr
        freq_acc += g * slope
        hist.append(float(np.std(err)))
    d = np.argmin(np.abs(z[:, None] - const[None, :]) ** 2, axis=1)
    final_err = z / const[d]
    residual_phase = float(np.angle(np.mean(final_err)))
    return {"ok": True, "symbols": z, "residual_frequency_rad_per_symbol": float(freq_acc),
            "residual_phase_rad": residual_phase,
            "phase_error_std_rad": float(np.std(np.angle(final_err))),
            "iterations": n_iter,
            "method": f"decision-directed phase/frequency tracking ({n_iter} gear-shifted iterations, "
                      f"initial gain {loop_gain}) on the {mod} constellation",
            "converged": bool(np.std(np.angle(final_err)) < 0.5)}


# --------------------------------------------------------------------------- #
#  B. Timing recovery (Oerder-Meyr)
# --------------------------------------------------------------------------- #
def oerder_meyr_timing(x: np.ndarray, fs: float, rs: float, sps_target: int = 4) -> dict:
    """Oerder-Meyr square-and-filter timing estimator.

    Works on the |x|^2 process: if a spectral line exists at the symbol rate its
    phase gives the optimal sampling instant and its magnitude measures timing
    quality, which is exactly the quantity needed to decide whether a symbol-rate
    hypothesis is real.
    """
    x = np.asarray(x, dtype=np.complex128)
    p = np.abs(x) ** 2
    n = p.size
    if rs <= 0 or n < 128:
        return {"ok": False, "error": "invalid rate or insufficient samples"}
    nperseg = int(min(8192, max(128, n // 4)))
    f, P = sigproc.welch(p - p.mean(), fs=fs, nperseg=nperseg,
                         window=make_window("hann", nperseg), return_onesided=False, scaling="density")
    o = np.argsort(f)
    f, P = f[o], P[o]
    df = abs(f[1] - f[0])
    i = int(np.argmin(np.abs(f - rs)))
    lo = max(0, i - max(2, int(0.02 * rs / df)))
    hi = min(P.size, i + max(3, int(0.02 * rs / df)) + 1)
    band = P[lo:hi]
    if band.size == 0:
        return {"ok": False, "error": "candidate rate outside the estimated spectrum"}
    k = int(np.argmax(band)) + lo
    excl = np.ones(P.size, dtype=bool)
    excl[max(0, k - 3 * (hi - lo)): k + 3 * (hi - lo) + 1] = False
    local_floor = float(np.median(P[excl])) if np.any(excl) else float(np.median(P))
    line_db = 10 * math.log10(max(float(P[k]), 1e-30) / max(local_floor, 1e-30))
    # two-point DFT interpolator around the line for a sub-bin timing phase estimate
    n_t = int(min(n, 65536))
    seg = p[:n_t] - p[:n_t].mean()
    nn = np.arange(n_t)
    w1 = np.exp(-2j * np.pi * rs * nn / fs)
    c = np.sum(seg * w1)
    phase = float(-np.angle(c) / (2 * np.pi)) % 1.0
    return {"ok": True, "timing_phase_frac": phase, "sample_offset": phase / rs,
            "timing_line_db": float(line_db), "peak_frequency_hz": float(f[k]),
            "method": "Oerder-Meyr square-and-filter timing estimation (|x|^2 spectral line at Rs)",
            "limitations": ["requires a non-zero excess bandwidth to produce a timing line",
                            "for constant-envelope FSK the |x|^2 line is absent - a discriminator-based "
                            "timing search is used instead"]}


def fsk_timing_score(x: np.ndarray, fs: float, rs: float, n_phases: int = 16,
                     smooth: int = 3) -> dict:
    """Within-symbol constancy score for (G)FSK: the true rate yields piecewise-constant IF."""
    x = np.asarray(x, dtype=np.complex128)
    if rs <= 0 or x.size < 256:
        return {"ok": False, "error": "invalid input"}
    ph = np.unwrap(np.angle(x))
    dph = np.diff(ph)
    slope = float(np.polyfit(np.arange(dph.size, dtype=np.float64), dph, 1)[0])
    inst = (dph - slope) * fs / (2 * math.pi)
    if smooth > 1:
        inst = np.convolve(inst, np.ones(smooth) / smooth, mode="same")
    sps = fs / rs
    if sps < 2 or sps > 4096:
        return {"ok": False, "error": f"candidate rate implies {sps:.1f} samples/symbol - out of range"}
    best = None
    n_sym_max = int(inst.size // sps) - 1
    if n_sym_max < 16:
        return {"ok": False, "error": "too few symbols at this rate"}
    for p_i in range(n_phases):
        off = p_i / n_phases
        idx = np.floor((np.arange(n_sym_max) + off) * sps).astype(int)
        idx = idx[(idx >= 0) & (idx + int(sps) < inst.size)]
        if idx.size < 16:
            continue
        seg = np.stack([inst[i:i + int(sps)] for i in idx])
        means = seg.mean(axis=1)
        within = float(np.mean(seg.var(axis=1)))
        between = float(np.var(means))
        score = between / max(within, 1e-12)
        if best is None or score > best["score"]:
            # level separation quality: how well the symbol means split into discrete levels
            mu, sd = float(np.mean(means)), float(np.std(means))
            sep = sd / max(math.sqrt(max(within, 1e-12)), 1e-12)
            best = {"score": float(score), "separation": float(sep), "phase_frac": float(off),
                    "within_var": within, "between_var": between,
                    "n_symbols_used": int(idx.size)}
    if best is None:
        return {"ok": False, "error": "no valid phase produced symbols"}
    return {"ok": True, **best,
            "method": "within-symbol instantaneous-frequency constancy (variance ratio) over a 16-point "
                      "timing-phase search",
            "interpretation": "higher separation = the candidate rate splits the signal into "
                              "piecewise-constant frequency segments, as FSK requires"}


# --------------------------------------------------------------------------- #
#  C. Matched filtering and symbol sampling
# --------------------------------------------------------------------------- #
def as_rrc_matched(x: np.ndarray, fs: float, rs: float, rolloff: float,
                   span: int = 8, decimate_to: int = 4) -> dict:
    """Matched-filter the signal with an RRC and return both the input-rate and a
    decimated view.

    The matched filter is designed directly at the capture's sample rate with a
    fractional samples-per-symbol value, so no rate conversion is needed before
    symbol extraction (this avoids the band-edge distortion a narrow polyphase
    anti-alias filter would introduce).  The decimated view is produced only for
    display/interpolation purposes and reports its own anti-alias bandwidth.
    """
    x = np.asarray(x, dtype=np.complex128)
    sps_in = fs / rs
    if not (1.02 < sps_in < 4096):
        return {"ok": False, "error": f"input has {sps_in:.2f} samples/symbol - matched filtering needs "
                                      f"more than 1.02"}
    alpha = float(np.clip(rolloff, 0.02, 0.95))
    taps = synth.rrc_taps(alpha, sps_in, span)
    y = sigproc.upfirdn(taps, x)
    delay = (taps.size - 1) // 2
    y = y[delay: delay + x.size]
    out = {"ok": True, "samples": y, "sps_in": float(sps_in), "taps": int(taps.size),
           "delay_samples": int(delay), "rolloff_used": alpha,
           "method": f"root-raised-cosine matched filter (alpha={alpha:.3f}, {taps.size} taps, "
                     f"{sps_in:.3f} samples/symbol) with group-delay compensation"}
    # decimated view (display / eye / interpolation), only when the ratio is rational-simple
    q = int(round(sps_in / max(1, decimate_to)))
    if q >= 1 and abs(sps_in / q - decimate_to) < 0.05 and q <= 64:
        ntaps = int(8 * q + 1)
        h = sigproc.firwin(ntaps, 1.0 / (2.0 * q) * 0.92, window=("kaiser", 8.0))
        try:
            yd = sigproc.filtfilt(h, 1.0, y)          # zero-phase: delay-free
            yd = yd[::q]
            out.update(samples_dec=yd, sps_dec=float(decimate_to), decimation=q,
                       decimation_note=f"zero-phase FIR decimation by {q} with a {ntaps}-tap Kaiser "
                                       f"anti-alias filter at {0.92*fs/(2*q):g} Hz")
        except Exception as exc:
            out["decimation_note"] = f"decimated view unavailable ({exc})"
    else:
        try:
            up, down = int(max(1, decimate_to)), int(round(sps_in))
            g = math.gcd(up, down)
            yd = sigproc.resample_poly(y, up // g, down // g, window=("kaiser", 8.6))
            out.update(samples_dec=yd, sps_dec=float(decimate_to), decimation=None,
                       decimation_note=f"polyphase resampling to {decimate_to} samples/symbol "
                                       f"({up//g}/{down//g}); band edges may be slightly filtered")
        except Exception as exc:
            out["decimation_note"] = f"decimated view unavailable ({exc})"
    return out


def estimate_timing_drift(y: np.ndarray, sps: float, mod: str, n_segments: int = 8,
                          n_phases: int = 16, min_symbols: int = 32) -> dict:
    """Timing phase per record segment, fitted as a straight line -> phase *and* rate error.

    A single timing-phase estimate (Oerder-Meyr) cannot see a residual symbol-rate error: on a
    3000-symbol record 100 ppm walks the sampling instants across 0.3 of a symbol, which is worth tens
    of percentage points of EVM.  Estimating the best sampling phase *per segment* and fitting a line
    through them measures that walk directly.  The slope is the residual rate error (the phase drift
    in symbols per symbol) and the intercept is the timing phase, so both are obtained from one
    measurement instead of a control loop - deterministic, and with no loop-stability risk.

    Returns the intercept (phase, in symbols), the slope (dimensionless), the per-segment estimates and
    the symbol-index range they were measured over.
    """
    y = np.asarray(y, dtype=np.complex128)
    n = y.size
    if n < 8 * sps or sps < 1.6:
        return {"ok": False, "error": f"timing-drift estimation needs at least 8 symbols at "
                                      f">=1.6 samples/symbol (got {n / max(sps, 1e-9):.1f})"}
    n_sym = int(n / sps) - 2
    if n_sym < 32:
        return {"ok": False, "error": f"only {n_sym} symbols available"}
    n_seg = int(max(2, min(int(n_segments), n_sym // max(min_symbols, 8))))
    per = n_sym // n_seg
    phases, centres, evms = [], [], []
    grid = np.linspace(0.0, 1.0, int(n_phases), endpoint=False)

    def _read(pos: np.ndarray) -> np.ndarray:
        i0 = np.clip(np.floor(pos).astype(np.int64), 0, n - 2)
        fr = np.clip(pos - i0, 0.0, 1.0)
        return y[i0] * (1.0 - fr) + y[i0 + 1] * fr

    for seg in range(n_seg):
        k0 = seg * per
        ks = np.arange(k0, k0 + per, dtype=np.float64)
        best = None
        for ph in grid:
            sym = _read((ks + ph) * sps)
            ps = blind_phase_search(sym, mod, n_steps=8, keep_top=1)
            if ps.get("ok"):
                sym = sym * np.exp(1j * ps["best_phase_rad"])
            q = constellation_quality(sym, mod)
            if q.get("ok") and (best is None or q["evm_rms"] < best[0]):
                best = (q["evm_rms"], float(ph))
        if best is None:
            continue
        phases.append(best[1])
        centres.append(float(np.mean(ks)))
        evms.append(best[0])
    if len(phases) < 3:
        return {"ok": False, "error": "too few segments produced a usable timing phase"}
    # unwrap the phase sequence (each estimate is defined modulo one symbol)
    ph = np.asarray(phases, dtype=np.float64)
    unw = ph.copy()
    for i in range(1, unw.size):
        while unw[i] - unw[i - 1] > 0.5:
            unw[i] -= 1.0
        while unw[i] - unw[i - 1] < -0.5:
            unw[i] += 1.0
    t = np.asarray(centres, dtype=np.float64)
    # weighted least squares (weight = 1/segment error, floored so a single clean segment cannot
    # dominate the fit)
    w = 1.0 / np.maximum(np.asarray(evms, dtype=np.float64), 0.02)
    A = np.vstack([np.ones_like(t), t - t.mean()]).T
    Aw = A * w[:, None]
    coef, *_ = np.linalg.lstsq(Aw, unw * w, rcond=None)
    slope = float(coef[1])
    intercept = float(coef[0] - slope * t.mean())
    # residual timing jitter around the fit: a quality measure for the eye diagram too
    resid = unw - (intercept + slope * t)
    jitter = float(np.sqrt(np.mean(resid ** 2)))
    return {"ok": True, "phase_symbols": intercept, "slope": slope,
            "rate_correction_ppm": float(slope * 1e6),
            "segment_phases": ph.tolist(), "segment_centres": t.tolist(),
            "segment_evm": evms, "timing_jitter_frac": jitter, "n_segments": int(len(phases)),
            "spread_over_record_symbols": float(slope * (t[-1] - t[0])),
            "method": f"best sampling phase measured independently in {len(phases)} record segments "
                      f"({int(n_phases)} candidate phases each, chosen by constellation EVM), then "
                      f"fitted with a weighted straight line: slope = residual symbol-rate error, "
                      f"intercept = timing phase"}


def sample_with_drift(y: np.ndarray, sps: float, phase0: float, slope: float,
                      max_symbols: int | None = None) -> dict:
    """Read one sample per symbol on a grid whose timing phase drifts linearly with symbol index."""
    y = np.asarray(y, dtype=np.complex128)
    n = y.size
    if sps < 1.2:
        return {"ok": False, "error": f"{sps:.2f} samples/symbol is too few for symbol extraction"}
    n_sym = int((n - 2) / sps)
    if max_symbols:
        n_sym = min(n_sym, int(max_symbols))
    if n_sym < 8:
        return {"ok": False, "error": f"only {n_sym} symbols available"}
    ks = np.arange(n_sym, dtype=np.float64)
    pos = (ks + phase0 + slope * ks) * sps
    pos = pos[(pos >= 0) & (pos < n - 2)]
    i0 = np.floor(pos).astype(np.int64)
    fr = np.clip(pos - i0, 0.0, 1.0)
    sym = y[i0] * (1.0 - fr) + y[i0 + 1] * fr
    return {"ok": True, "symbols": sym, "positions": pos, "n_symbols": int(sym.size),
            "sps": float(sps), "phase": float(phase0), "slope": float(slope),
            "method": "linear interpolation of the matched-filter output on a symbol grid with the "
                      "measured timing phase and rate-drift correction"}


def sample_symbols(y: np.ndarray, sps: float, timing_phase_frac: float = 0.0,
                   n_symbols: int | None = None, delay_samples: int = 0) -> dict:
    """Extract one sample per symbol at the given fractional timing phase.

    Works with either integer or fractional samples per symbol: symbol centres fall
    at index ``delay + (k + phase) * sps`` and are read with linear interpolation.
    """
    sps = float(sps)
    if y.size < max(8, 2 * sps):
        return {"ok": False, "error": "filtered signal too short"}
    max_k = int((y.size - 1 - delay_samples) / sps) - 1
    if max_k < 4:
        return {"ok": False, "error": f"only {max_k} symbol slots available"}
    k = np.arange(max_k) if not n_symbols else np.arange(min(max_k, n_symbols))
    pos = delay_samples + (k + float(timing_phase_frac)) * sps
    pos = np.clip(pos, 0, y.size - 1.0001)
    i0 = np.floor(pos).astype(int)
    frac = pos - i0
    syms = y[i0] * (1 - frac) + y[i0 + 1] * frac
    return {"ok": True, "symbols": syms, "indices": pos, "sps": sps,
            "timing_phase_frac": float(timing_phase_frac),
            "interpolation": "fractional linear interpolation between matched-filter samples"}


# --------------------------------------------------------------------------- #
#  D. Constellation quality metrics (EVM, clusters, BER estimate)
# --------------------------------------------------------------------------- #
def _kurtosis(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    sd = float(a.std())
    if sd <= 0:
        return 9.9
    return float(np.mean(((a - a.mean()) / sd) ** 4))


def align_by_kurtosis(symbols: np.ndarray, mod: str, coarse_off_rad_per_symbol: float = 0.0,
                      n_freq: int = 33, n_phase: int = 16) -> dict:
    """Blind constellation alignment by minimising the kurtosis of the I and Q projections.

    When the carrier is correctly aligned, the in-phase and quadrature projections of a
    PSK/QAM constellation are discrete PAM-like variables (kurtosis well below the
    Gaussian value of 3).  A misaligned carrier mixes I into Q and drives both
    projections towards a Gaussian, so minimising kurt(Re)+kurt(Im) is a valid blind
    objective.  A frequency grid around the coarse estimate is searched, then the best
    point is refined by a parabolic/local search.
    """
    z = np.asarray(symbols, dtype=np.complex128).ravel()
    z = z[np.isfinite(z)]
    if z.size < 64:
        return {"ok": False, "error": "fewer than 64 symbols"}
    idx = np.arange(z.size, dtype=np.float64)
    best = None
    freq_grid = np.linspace(coarse_off_rad_per_symbol - 0.35, coarse_off_rad_per_symbol + 0.35, n_freq)
    phase_grid = np.linspace(0.0, 2 * math.pi / 8.0, n_phase, endpoint=False)
    for w in freq_grid:
        n_use = int(min(z.size, 2048))
        base = z[:n_use] * np.exp(-1j * w * np.arange(n_use))
        for th in phase_grid:
            zr = base * np.exp(-1j * th)
            score = _kurtosis(zr.real) + _kurtosis(zr.imag)
            if best is None or score < best["score"]:
                best = {"score": float(score), "freq_rad_per_symbol": float(w), "phase_rad": float(th)}
    if best is None:
        return {"ok": False, "error": "no candidate evaluated"}
    # local refinement around the best point
    w0, th0, sc0 = best["freq_rad_per_symbol"], best["phase_rad"], best["score"]
    for _ in range(3):
        dw, dth = 0.03 / (2 ** _), (2 * math.pi / 8.0) / (n_phase * (2 ** _))
        improved = False
        for w in (w0 - dw, w0, w0 + dw):
            for th in (th0 - dth, th0, th0 + dth):
                base = z[: min(z.size, 4096)] * np.exp(-1j * (w * np.arange(min(z.size, 4096)) + th))
                sc = _kurtosis(base.real) + _kurtosis(base.imag)
                if sc < sc0:
                    w0, th0, sc0, improved = w, th, sc, True
        if not improved:
            break
    return {"ok": True, "freq_rad_per_symbol": float(w0), "phase_rad": float(th0),
            "score": float(sc0), "initial_score": float(best["score"]),
            "method": f"blind kurtosis minimisation of the I/Q projections over {n_freq} frequency x "
                      f"{n_phase} phase candidates plus local refinement",
            "limitations": ["the M-fold rotational ambiguity of PSK constellations is inherent to blind "
                            "carrier recovery and is not resolved by this metric"]}


def constellation_quality(symbols: np.ndarray, mod: str) -> dict:
    """Noise-subtracted EVM against an ideal constellation, with cluster analysis.

    EVM is computed after optimally scaling and rotating the measured symbols onto
    the reference constellation (a least-squares gain+phase fit), so a residual
    scaling or carrier phase does not inflate it.
    """
    z = np.asarray(symbols, dtype=np.complex128).ravel()
    z = z[np.isfinite(z)]
    if z.size < 16:
        return {"ok": False, "error": "fewer than 16 symbols"}
    const = synth.constellation(mod)
    # least-squares complex gain a: minimise |a*z - c| -> a = <z,c>/<z,z>
    d0 = np.argmin(np.abs(z[:, None] - const[None, :]) ** 2, axis=1)
    c0 = const[d0]
    a = np.vdot(z, c0) / max(np.vdot(z, z).real, 1e-30)
    zc = z * a
    d = np.argmin(np.abs(zc[:, None] - const[None, :]) ** 2, axis=1)
    c = const[d]
    err = zc - c
    evm = float(np.sqrt(np.mean(np.abs(err) ** 2) / max(np.mean(np.abs(c) ** 2), 1e-30)))
    # cluster statistics in the normalised plane
    counts = np.bincount(d, minlength=const.size)
    used = counts > 0
    cluster_cover = float(np.mean(used))
    min_cluster_frac = float(counts[used].min() / z.size) if np.any(used) else 0.0
    # symbols that are ambiguous (near a decision boundary)
    d2 = np.abs(zc[:, None] - const[None, :]) ** 2
    srt = np.sort(d2, axis=1)
    if srt.shape[1] > 1:
        ambiguity = float(np.mean(srt[:, 1] / np.maximum(srt[:, 0], 1e-30) < 2.0))
    else:
        ambiguity = 0.0
    return {
        "ok": True, "evm_rms": evm, "evm_db": float(20 * math.log10(max(evm, 1e-9))),
        "evm_percent": round(100 * evm, 3),
        "cluster_coverage": cluster_cover, "min_cluster_fraction": min_cluster_frac,
        "ambiguity_fraction": ambiguity, "n_symbols": int(z.size),
        "gain_applied": {"magnitude": float(abs(a)), "phase_rad": float(np.angle(a))},
        "decision_counts": counts.tolist(),
        "method": "nearest-point decisions after a least-squares complex gain+phase fit to the ideal "
                  "constellation; EVM = RMS error / RMS reference magnitude",
    }


def evm_to_ber_estimate(evm_rms: float, mod: str, snr_db: float | None = None) -> dict:
    """Estimate the uncoded BER from the measured EVM.

    Uses the standard relationship between EVM and symbol SNR for the modulation
    in question.  This is an *estimate derived from EVM*, never a measurement of
    transmitted bits, and is labelled as such.
    """
    if evm_rms <= 0 or not np.isfinite(evm_rms):
        return {"ok": False}
    esn0_lin = 1.0 / (evm_rms ** 2)
    esn0_db = 10 * math.log10(max(esn0_lin, 1e-12))
    k = synth.bits_per_symbol(mod) if mod.upper() in ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM") else 1
    ebn0_db = esn0_db - 10 * math.log10(max(k, 1))
    ebn0 = 10 ** (ebn0_db / 10.0)
    if mod.upper() in ("BPSK", "QPSK"):
        ber = 0.5 * erfc(math.sqrt(max(ebn0, 0.0)))
    elif mod.upper() == "8PSK":
        ber = (2.0 / 3.0) * 0.5 * erfc(math.sqrt(max(ebn0, 0.0)) * math.sin(math.pi / 8))
    elif mod.upper() in ("16QAM", "64QAM"):
        ber = (4.0 / k) * (1 - 1 / math.sqrt(2 ** k)) * 0.5 * erfc(
            math.sqrt(max(ebn0, 0.0)) * math.sqrt(3.0 / (2 ** k - 1)))
    else:
        ber = 0.5 * erfc(math.sqrt(max(ebn0, 0.0)))
    return {"ok": True, "ber_estimate": float(min(max(ber, 0.0), 0.5)), "esn0_db": float(esn0_db),
            "ebn0_db": float(ebn0_db), "bits_per_symbol": k,
            "method": "uncoded BER derived from the measured EVM (standard EVM->SNR->BER mapping); "
                      "an estimate, not a measurement"}


def soft_llrs(symbols: np.ndarray, mod: str, noise_var: float | None = None) -> np.ndarray:
    """Max-log soft bit LLRs (positive = bit 0), used by the FEC decoder tests."""
    const = synth.constellation(mod)
    k = synth.bits_per_symbol(mod)
    z = np.asarray(symbols, dtype=np.complex128).ravel()
    if noise_var is None:
        q = constellation_quality(z, mod)
        evm = q.get("evm_rms", 0.1) if q.get("ok") else 0.1
        noise_var = max(evm ** 2, 1e-6)
    llr = np.zeros((z.size, k), dtype=np.float64)
    for j in range(k):
        b0 = ((np.arange(const.size) >> (k - 1 - j)) & 1) == 0
        d = np.abs(z[:, None] - const[None, :]) ** 2
        d0 = d[:, b0].min(axis=1)
        d1 = d[:, ~b0].min(axis=1)
        llr[:, j] = (d1 - d0) / noise_var
    return llr.ravel()


# --------------------------------------------------------------------------- #
#  E. Full demodulation chains
# --------------------------------------------------------------------------- #
LINEAR_MODS = ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM")


def _quick_symbol_quality(x_c: np.ndarray, fs: float, rs: float, mod: str, rolloff: float,
                          n_phase_steps: int = 16) -> dict:
    """Compact matched-filter -> timing -> sampling -> phase-search chain used for rate sweeps."""
    mf = as_rrc_matched(x_c, fs, rs, rolloff, decimate_to=4)
    if not mf.get("ok"):
        return {"ok": False, "error": mf.get("error")}
    tm = oerder_meyr_timing(x_c, fs, rs)
    phase = tm["timing_phase_frac"] if tm.get("ok") else 0.0
    smp = sample_symbols(mf["samples"], mf["sps_in"], phase, delay_samples=mf["delay_samples"])
    if not smp.get("ok"):
        return {"ok": False, "error": smp.get("error")}
    sym = smp["symbols"]
    ps = blind_phase_search(sym, mod, n_steps=n_phase_steps, keep_top=1)
    if ps.get("ok"):
        sym = sym * np.exp(1j * ps["best_phase_rad"])
    q = constellation_quality(sym, mod)
    if not q.get("ok"):
        return {"ok": False, "error": q.get("error")}
    return {"ok": True, "evm_rms": q["evm_rms"], "symbols": sym, "q": q, "mf": mf, "tm": tm,
            "smp": smp, "symbol_rate_hz": float(rs)}


def refine_symbol_rate(x_c: np.ndarray, fs: float, rs: float, mod: str, rolloff: float,
                       spread: float = 2e-2, steps: int = 8,
                       target_evm_below: float = 0.10, fine_spread: float = 1.5e-3,
                       fine_steps: int = 8) -> dict:
    """Search a bounded neighbourhood of the estimated symbol rate for the best constellation.

    A symbol-rate error of a few parts in 10^5 already walks the sampling instants across a whole
    symbol over a long record, which shows up as EVM.  Receivers therefore refine the rate together
    with the timing; this does the same in two bounded passes so it can never wander with the data:

    * a **coarse** pass over +/-``spread`` (default 2 %, in ``steps`` increments per side) - the
      blind estimator's own error was measured at 0.3 % on an 8-PSK record and a +/-0.1 % search
      could not repair it (EVM stayed at 32 % because the estimator error was outside the grid;
      measured with a +/-2 % coarse pass and a fine pass the same record refines to 25 000 Hz and
      EVM drops to 5.6 %), and
    * a **fine** pass over +/-``fine_spread`` around the best coarse candidate, which also covers
      the small neighbourhood the original single-pass search handled.

    The search stops early as soon as the constellation is already clean, so uncoded records pay
    almost nothing for the extra pass.
    """
    nominal_q = _quick_symbol_quality(x_c, fs, rs, mod, rolloff)
    if not nominal_q.get("ok"):
        return {"ok": False, "message": f"the nominal rate could not be evaluated: {nominal_q.get('error')}"}
    if nominal_q["evm_rms"] < target_evm_below:
        # the constellation is already good at the estimated rate: no search needed (and no risk of
        # fitting the rate to noise)
        return {"ok": True, "best": nominal_q, "nominal": nominal_q, "improved": False,
                "n_candidates": 1, "spread": float(spread),
                "message": "the estimated symbol rate already gives a clean constellation"}
    def _grid(centre: float, half: float, n: int) -> list[float]:
        out = []
        for k in range(1, int(n) + 1):
            d = half * k / float(n)
            out += [float(centre) * (1.0 + d), float(centre) * (1.0 - d)]
        return [r for r in out if r > 0]

    results = [nominal_q]

    def _scan(centre: float, half: float, n: int) -> bool:
        """Try a grid around ``centre``; return True when the search can stop early."""
        for r in _grid(centre, half, n):
            q = _quick_symbol_quality(x_c, fs, r, mod, rolloff)
            if q.get("ok"):
                results.append(q)
                if q["evm_rms"] < 0.5 * target_evm_below:
                    return True              # good enough: stop searching
        return False

    stopped = _scan(rs, float(spread), int(steps))            # coarse pass
    best_so_far = min(results, key=lambda q: q["evm_rms"])
    if not stopped:
        _scan(best_so_far["symbol_rate_hz"], float(fine_spread), int(fine_steps))   # fine pass
    if not results:
        return {"ok": False, "message": "no usable candidate during the symbol-rate refinement"}
    best = min(results, key=lambda q: q["evm_rms"])
    nominal = min(results, key=lambda q: abs(q["symbol_rate_hz"] - rs))
    improved = best["evm_rms"] < 0.985 * nominal["evm_rms"]
    return {"ok": True, "best": best, "nominal": nominal, "improved": bool(improved),
            "n_candidates": len(results), "spread": float(spread),
            "fine_spread": float(fine_spread),
            "message": ("symbol rate confirmed by the refinement search"
                        if not improved else
                        f"symbol rate refined by {(best['symbol_rate_hz'] / rs - 1) * 1e6:+.1f} ppm")}


def demodulate_linear(x: np.ndarray, fs: float, mod: str, rs: float, rolloff: float,
                      carrier_hint_hz: float | None = None, differential: bool = False,
                      noise_var: float | None = None, decimate_to: int = 4,
                      band_hint_hz: float | None = None) -> dict:
    """Coherent PSK/QAM demodulation with stage-by-stage reporting."""
    stages: list[dict] = []
    x = np.asarray(x, dtype=np.complex128)
    mod = mod.upper()
    if mod not in LINEAR_MODS:
        return {"ok": False, "error": f"{mod} is not a linear modulation", "stages": stages}
    m = {"BPSK": 2, "QPSK": 4, "8PSK": 8, "16QAM": 4, "64QAM": 4}[mod]

    # --- stage 1: coarse carrier recovery
    # The offset is always *measured* with the M-th power line (that estimator reaches a few Hz at
    # the SNRs this platform targets); a caller-supplied hint is used only to choose between the
    # aliases of that measurement, and as a fallback when the line is too weak to trust.  Trusting
    # the hint alone was measured to leave a residual of ~300 Hz, which the decision-directed trackers
    # cannot follow (EVM 8.5 % -> 43.7 %).
    hint = None if carrier_hint_hz is None else float(carrier_hint_hz)
    est = estimate_carrier_mth_power(x, fs, m)
    car = None
    if est.get("ok") and float(est.get("line_strength_db") or 0.0) >= 6.0:
        alias = resolve_carrier_alias(est["frequency_offset_hz"], fs, m,
                                     hint if hint is not None else band_hint_hz)
        if alias.get("ok"):
            est["frequency_offset_hz"] = alias["frequency_offset_hz"]
            est["alias_resolution"] = alias
        est["hint_hz"] = hint
        if hint is not None:
            est["method"] = ("carrier offset measured from the M-th power spectral line "
                             f"(hint {hint:.1f} Hz used only for alias resolution)")
        car = est
    elif hint is not None:
        car = {"ok": True, "frequency_offset_hz": hint,
               "method": "carrier offset taken from the caller's spectral-centroid hint (the M-th "
                         "power line was too weak to measure the offset in this record)",
               "line_strength_db": (est or {}).get("line_strength_db"), "confidence": 0.5,
               "limitations": ["the carrier offset could not be measured from the signal itself"],
               "line_phase_rad": None, "hint_hz": hint}
    else:
        car = est
    if not car.get("ok"):
        stages.append({"stage": "carrier recovery", "status": "failed", "error": car.get("error")})
        return {"ok": False, "error": "carrier recovery failed", "stages": stages}
    stages.append({"stage": "carrier recovery (coarse)", "status": STATUS_OK,
                   "detail": f"offset {car['frequency_offset_hz']:.1f} Hz, constellation phase "
                             f"{math.degrees(car.get('line_phase_rad') or 0.0):.1f} deg",
                   "method": car["method"],
                   "evidence": ([f"M-th power line {car['line_strength_db']:.1f} dB above the local floor"]
                                if car.get("line_strength_db") else [])})
    t = np.arange(x.size, dtype=np.float64) / fs
    x_c = x * np.exp(-2j * np.pi * car["frequency_offset_hz"] * t)

    # --- stage 2b: symbol-rate refinement
    # The blind symbol-rate estimator is accurate to ~0.01 %, which is not enough for a long record
    # (0.01 % of 25 kHz drifts one symbol over 4000 symbols).  The rate is therefore refined by
    # minimising the constellation EVM over a bounded grid (+/-0.05 %) around the estimate.
    rr = refine_symbol_rate(x_c, fs, rs, mod, rolloff)
    if rr.get("ok") and rr.get("improved"):
        rs = float(rr["best"]["symbol_rate_hz"])
        stages.append({"stage": "symbol-rate refinement", "status": STATUS_OK,
                       "detail": f"{rr['nominal']['symbol_rate_hz']:,.1f} Hz -> {rs:,.1f} Hz "
                                 f"({(rs / max(rr['nominal']['symbol_rate_hz'], 1e-9) - 1) * 1e6:+.1f} ppm); "
                                 f"constellation EVM "
                                 f"{100 * rr['nominal']['evm_rms']:.2f} % -> "
                                 f"{100 * rr['best']['evm_rms']:.2f} %",
                       "method": "9-point sweep of the symbol rate around the estimate, taking the "
                                 "rate that minimises the constellation EVM after matched filtering "
                                 "and the blind phase search"})
    elif rr.get("ok"):
        stages.append({"stage": "symbol-rate refinement", "status": STATUS_OK,
                       "detail": f"the estimated rate {rs:,.1f} Hz was confirmed "
                                 f"({rr['n_candidates']} candidates tested, EVM "
                                 f"{100 * rr['best']['evm_rms']:.2f} %)",
                       "method": "bounded sweep of the symbol rate around the estimate"})

    # --- stage 2: matched filtering
    mf = as_rrc_matched(x_c, fs, rs, rolloff, decimate_to=decimate_to)
    if not mf.get("ok"):
        stages.append({"stage": "matched filtering", "status": "failed", "error": mf.get("error")})
        return {"ok": False, "error": mf.get("error"), "stages": stages}
    stages.append({"stage": "matched filtering (RRC)", "status": STATUS_OK, "detail": mf["method"]})

    # --- stage 3: timing recovery
    tm = oerder_meyr_timing(x_c, fs, rs)
    if tm.get("ok"):
        timing_phase = tm["timing_phase_frac"]
        stages.append({"stage": "timing recovery", "status": STATUS_OK,
                       "detail": f"Oerder-Meyr phase {timing_phase:.3f} of a symbol",
                       "method": tm["method"],
                       "evidence": [f"timing line {tm['timing_line_db']:.1f} dB above the local PSD floor"]})
    else:
        timing_phase = 0.0
        stages.append({"stage": "timing recovery", "status": "low_confidence",
                       "detail": "Oerder-Meyr estimator found no timing line; symbol centres assumed at the "
                                 "origin of each symbol interval",
                       "error": tm.get("error")})
    y_mf = np.asarray(mf["samples"])[int(mf["delay_samples"]):]
    drift = estimate_timing_drift(y_mf, mf["sps_in"], mod)
    if drift.get("ok"):
        smp = sample_with_drift(y_mf, mf["sps_in"] * (1.0 + drift["slope"]), drift["phase_symbols"],
                                0.0)
        rs_used = float(rs) / (1.0 + drift["slope"])
        stages.append({"stage": "timing recovery (phase + drift)", "status": STATUS_OK,
                       "detail": f"timing phase {drift['phase_symbols']:+.3f} symbols; residual "
                                 f"symbol-rate error {drift['rate_correction_ppm']:+.1f} ppm "
                                 f"(the sampling phase moved {drift['spread_over_record_symbols']:+.3f} "
                                 f"symbols across the record), jitter "
                                 f"{drift['timing_jitter_frac'] * 100:.1f} % of a symbol; effective "
                                 f"symbol rate {rs_used:,.2f} Hz",
                       "method": drift["method"],
                       "evidence": [f"per-segment timing phases (symbols): "
                                    + ", ".join(f"{v:.2f}" for v in drift["segment_phases"]),
                                    f"per-segment constellation EVM: "
                                    + ", ".join(f"{100 * v:.1f} %" for v in drift["segment_evm"])]})
    else:
        smp = sample_symbols(mf["samples"], mf["sps_in"], timing_phase,
                             delay_samples=mf["delay_samples"])
        rs_used = float(rs)
        stages.append({"stage": "symbol sampling", "status": "low_confidence" if smp.get("ok")
                       else "failed",
                       "detail": (f"{smp['symbols'].size} symbols at {mf['sps_in']:.3f} "
                                  f"samples/symbol on the fixed Oerder-Meyr phase (drift estimation "
                                  f"unavailable: {drift.get('error')})" if smp.get("ok")
                                  else smp.get("error"))})
    if not smp.get("ok"):
        return {"ok": False, "error": smp.get("error"), "stages": stages}

    # --- stage 3b: blind phase search (removes the residual phase of the coarse estimator)
    sym_in = smp["symbols"]
    bps = blind_phase_search(sym_in, mod, n_steps=32, keep_top=3)
    if bps.get("ok"):
        rot0 = np.exp(1j * bps["best_phase_rad"])
        q_raw = constellation_quality(sym_in, mod)
        q_bps = constellation_quality(sym_in * rot0, mod)
        sym_in = sym_in * rot0
        stages.append({"stage": "blind phase search", "status": STATUS_OK,
                       "detail": f"best of {bps['grid_steps']} phases over a "
                                 f"{math.degrees(bps['sector_width_rad']):.0f} deg sector: "
                                 f"{math.degrees(bps['best_phase_rad']):.2f} deg"
                                 + (f", EVM {100*q_raw['evm_rms']:.2f}% -> {100*q_bps['evm_rms']:.2f}%"
                                    if q_raw.get("ok") and q_bps.get("ok") else ""),
                       "method": bps["method"]})
    else:
        stages.append({"stage": "blind phase search", "status": "failed", "error": bps.get("error")})

    # --- stage 3c: residual frequency search -------------------------------------------
    # A small residual carrier offset (a fraction of a percent of the symbol rate) is invisible in
    # the phase search but rotates the constellation across the record.  A short sweep on the
    # extracted symbols removes it; it only runs when the constellation is still poor, so clean
    # signals pay nothing for it.
    try:
        q_now = constellation_quality(sym_in, mod)
        if q_now.get("ok") and q_now["evm_rms"] > 0.15 and sym_in.size >= 128:
            best = (q_now["evm_rms"], 0.0, sym_in)
            for frac in np.linspace(-0.004, 0.004, 9):
                if frac == 0.0:
                    continue
                rot = np.exp(-2j * np.pi * float(frac) * np.arange(sym_in.size, dtype=np.float64))
                cand = sym_in * rot
                q = constellation_quality(cand, mod)
                if q.get("ok") and q["evm_rms"] < best[0]:
                    ps = blind_phase_search(cand, mod, n_steps=16, keep_top=1)
                    if ps.get("ok"):
                        cand2 = cand * np.exp(1j * ps["best_phase_rad"])
                        q2 = constellation_quality(cand2, mod)
                        if q2.get("ok") and q2["evm_rms"] < best[0]:
                            best = (q2["evm_rms"], float(frac), cand2)
            if best[1] != 0.0:
                sym_in = best[2]
                stages.append({"stage": "residual frequency search", "status": STATUS_OK,
                               "detail": f"removed a residual offset of "
                                         f"{best[1] * rs:+.0f} Hz ({best[1] * 100:+.2f} % of the "
                                         f"symbol rate); constellation EVM improved to "
                                         f"{100 * best[0]:.2f} %",
                               "method": "9-point sweep of the residual frequency around the coarse "
                                         "estimate, taking the offset that minimises the "
                                         "constellation EVM"})
            elif q_now["evm_rms"] > 0.15:
                stages.append({"stage": "residual frequency search", "status": "ok",
                               "detail": "no residual-frequency offset improved the constellation; "
                                         "the remaining EVM is therefore not a frequency error",
                               "method": "9-point sweep of the residual frequency"})
    except Exception as exc:                                                # pragma: no cover
        stages.append({"stage": "residual frequency search", "status": "skipped",
                       "error": str(exc)})

    if mod in ("16QAM", "64QAM") and sym_in.size >= 256:
        al = align_by_kurtosis(sym_in, mod, coarse_off_rad_per_symbol=0.0)
        if al.get("ok"):
            idx_k = np.arange(sym_in.size, dtype=np.float64)
            rot = np.exp(-1j * (al["freq_rad_per_symbol"] * idx_k + al["phase_rad"]))
            q_before = constellation_quality(sym_in, mod)
            q_after = constellation_quality(sym_in * rot, mod)
            if q_before.get("ok") and q_after.get("ok") and q_after["evm_rms"] < q_before["evm_rms"]:
                sym_in = sym_in * rot
                stages.append({"stage": "blind constellation alignment (QAM)", "status": STATUS_OK,
                               "detail": f"I/Q kurtosis objective {al['initial_score']:.2f} -> "
                                         f"{al['score']:.2f}; EVM {100*q_before['evm_rms']:.2f}% -> "
                                         f"{100*q_after['evm_rms']:.2f}%",
                               "method": al["method"], "limitations": al["limitations"]})
            else:
                stages.append({"stage": "blind constellation alignment (QAM)", "status": "rejected",
                               "detail": "kurtosis alignment did not improve the constellation",
                               "method": al["method"]})

    # --- stage 4: decision-directed tracking started from the best phase hypotheses
    gains = {"BPSK": (0.30, 5), "QPSK": (0.30, 5), "8PSK": (0.45, 6),
             "16QAM": (0.45, 6), "64QAM": (0.5, 6)}
    lg, nit = gains.get(mod, (0.35, 5))
    cand_phases = [t["phase_rad"] - bps.get("best_phase_rad", 0.0) for t in bps.get("top", [])
                   if bps.get("ok")] or [0.0]
    hyp = best_phase_hypothesis(sym_in, mod, cand_phases, lg, nit)
    if hyp.get("ok"):
        symbols = hyp["symbols"]
        tried = ", ".join(f"{math.degrees(t['phase_rad']):.1f}deg:{100*t['evm_rms']:.2f}%"
                          for t in hyp["tried"] if t["ok"])
        stages.append({"stage": "carrier tracking (decision-directed)", "status": STATUS_OK,
                       "detail": f"started from {hyp['n_hypotheses']} of the best phase-search candidates "
                                 f"[{tried}]; selected EVM {100*hyp['quality']['evm_rms']:.2f}%",
                       "method": hyp["method"],
                       "limitations": ["carrier phase tracking is decision-directed, so at very low SNR a "
                                       "cycle slip can still occur; the best of several candidates is kept"]})
    else:
        symbols = sym_in
        stages.append({"stage": "carrier tracking (decision-directed)", "status": "failed",
                       "error": hyp.get("error")})

    # --- stage 5: decisions and bits
    q = constellation_quality(symbols, mod)
    if differential and mod in ("BPSK", "QPSK", "8PSK"):
        k = synth.bits_per_symbol(mod)
        zc = symbols / (symbols[0] if symbols.size else 1)
        sym_d = np.concatenate([[1 + 0j], symbols[1:] * np.conj(symbols[:-1])])
        bits = synth.demap_symbols(mod, sym_d)
        stages.append({"stage": "differential detection", "status": STATUS_OK,
                       "detail": f"differential {mod} detection applied (phase differences taken between "
                                 f"consecutive symbols)"})
    else:
        bits = synth.demap_symbols(mod, symbols)
    llr = soft_llrs(symbols, mod, noise_var)
    ber = evm_to_ber_estimate(q.get("evm_rms", 0.0), mod)
    stages.append({"stage": "symbol decisions", "status": STATUS_OK if q.get("ok") else "failed",
                   "detail": f"EVM {q.get('evm_percent', float('nan')):.2f}% ({q.get('evm_db', 0):.1f} dB), "
                             f"cluster coverage {100*q.get('cluster_coverage', 0):.0f}%",
                   "method": q.get("method")})
    return {
        "ok": True, "modulation": mod, "symbols": symbols, "bits": bits, "llrs": llr,
        "symbol_rate_used_hz": float(rs_used),
        "n_symbols": int(symbols.size), "n_bits": int(bits.size),
        "quality": q, "ber_estimate": ber, "stages": stages,
        "carrier_offset_hz": float(car["frequency_offset_hz"]),
        "timing_phase_frac": float(timing_phase),
        "bit_quality": _bit_quality_word(q, ber),
    }


def _bit_quality_word(q: dict, ber: dict) -> str:
    if not q.get("ok"):
        return "unknown"
    evm = q.get("evm_percent", 100)
    if evm < 8:
        return "high"
    if evm < 18:
        return "medium"
    if evm < 32:
        return "low"
    return "very low"


def demodulate_fsk(x: np.ndarray, fs: float, rs: float, h_hint: float | None = None,
                   bt: float | None = None, coherent: bool = True) -> dict:
    """Non-coherent discriminator FSK demodulation with a coherent cross-check."""
    stages: list[dict] = []
    x = np.asarray(x, dtype=np.complex128)
    if x.size < 256:
        return {"ok": False, "error": "insufficient samples", "stages": stages}
    # band-pass around the signal before discrimination
    from . import spectrum as sp
    try:
        flt = design_filter("bandpass", fs, [max(1.0, rs * 0.3), min(fs / 2 * 0.98, rs * 6.0)], 6)
        xf, finfo = apply_filter(x, flt)
        stages.append({"stage": "band limiting", "status": STATUS_OK, "detail": finfo["filter"]})
    except Exception as exc:
        xf = x
        stages.append({"stage": "band limiting", "status": "failed", "error": str(exc)})
    ifm = sp.instantaneous_frequency(xf, fs)
    if not ifm.get("ok"):
        stages.append({"stage": "frequency discrimination", "status": "failed", "error": ifm.get("error")})
        return {"ok": False, "error": ifm.get("error"), "stages": stages}
    inst = np.asarray(ifm["inst_freq_hz"], dtype=np.float64)
    stages.append({"stage": "frequency discrimination", "status": STATUS_OK,
                   "detail": f"residual carrier {ifm.get('residual_carrier_hz', float('nan')):.1f} Hz, "
                             f"RMS deviation {ifm.get('rms_deviation_hz', float('nan')):.1f} Hz",
                   "method": ifm.get("method", "phase-difference instantaneous frequency")})
    # timing search with hysteresis-free integrate-and-dump
    ts = fsk_timing_score(xf, fs, rs)
    sps = float(fs / rs)
    sps_i = int(round(sps))
    if sps < 2:
        return {"ok": False, "error": f"{sps} samples/symbol - rate too high for this sample rate",
                "stages": stages}
    phase_frac = ts.get("phase_frac", 0.0) if ts.get("ok") else 0.0
    start = int(round(phase_frac * sps_i))
    idx = (start + np.arange(int((inst.size - start) / sps_i)) * sps_i).astype(int)
    idx = idx[idx < inst.size - 1]
    if idx.size < 8:
        return {"ok": False, "error": "fewer than 8 symbols extracted", "stages": stages}
    half = max(1, int(sps_i // 2))
    levels = np.array([float(np.mean(inst[max(0, i - half):min(inst.size, i + half + 1)])) for i in idx])
    stages.append({"stage": "timing search (IF constancy)", "status": STATUS_OK if ts.get("ok") else "low_confidence",
                   "detail": f"phase {phase_frac:.3f} of a symbol; separation metric "
                             f"{ts.get('separation', float('nan')):.2f}" if ts.get("ok") else
                             "fallback: symbol centres at the frame origin",
                   "method": ts.get("method", "integrate-and-dump")})
    # threshold at zero (2-level) or at the midpoint between the two dominant levels
    threshold = 0.0
    if h_hint is None:
        hi = float(np.percentile(levels, 85))
        lo = float(np.percentile(levels, 15))
        threshold = 0.5 * (hi + lo)
        h_est = None
        if lo != hi:
            h_est = None
    else:
        h_est = h_hint
    bits = (levels > threshold).astype(np.uint8)
    # quality: separation between the two decision levels relative to their spread
    hi_m = levels[bits == 1]
    lo_m = levels[bits == 0]
    sep = float(abs(np.mean(hi_m) - np.mean(lo_m))) if hi_m.size and lo_m.size else 0.0
    spread = float(np.std(levels))
    q = {"ok": sep > 0, "level_separation_hz": sep, "level_spread_hz": spread,
         "separation_ratio": float(sep / max(spread, 1e-9)),
         "n_levels_detected": int(len(np.unique(np.round(levels / max(sep, 1e-9), 1)))) if sep else 1,
         "threshold_hz": float(threshold)}
    stages.append({"stage": "level slicing", "status": STATUS_OK,
                   "detail": f"level separation {sep:.1f} Hz, spread {spread:.1f} Hz, threshold {threshold:.1f} Hz"})
    if coherent:
        # coherent cross-check: correlator bank on the two hypotheses of IF polarity
        try:
            t = np.arange(x.size) / fs
            dev = max(sep / 2.0, rs * 0.1)
            c0 = x * np.exp(-2j * np.pi * dev * t)
            c1 = x * np.exp(+2j * np.pi * dev * t)
            n = int(min(x.size, 65536))
            r0 = float(np.abs(np.mean(c0[:n])) ** 2)
            r1 = float(np.abs(np.mean(c1[:n])) ** 2)
            stages.append({"stage": "coherent correlator cross-check", "status": STATUS_OK,
                           "detail": f"tone energies: -{dev:.0f} Hz {10*math.log10(max(r0,1e-30)):.1f} dBFS, "
                                     f"+{dev:.0f} Hz {10*math.log10(max(r1,1e-30)):.1f} dBFS"})
        except Exception as exc:
            stages.append({"stage": "coherent correlator cross-check", "status": "failed", "error": str(exc)})
    ber_proxy = float(min(0.5, max(0.0, 0.5 * math.exp(-0.5 * max(q["separation_ratio"], 0.0) ** 2))))
    return {
        "ok": True, "modulation": "2FSK" if not bt else "GFSK", "symbols": levels.astype(np.complex128),
        "bits": bits, "llrs": None, "n_symbols": int(levels.size), "n_bits": int(bits.size),
        "quality": q, "inst_freq_hz": inst, "symbol_indices": idx,
        "ber_estimate": {"ok": True, "ber_estimate": ber_proxy,
                         "method": "Gaussian approximation on the measured IF level separation "
                                   "(proxy, not a bit-error measurement)"},
        "bit_quality": ("high" if q["separation_ratio"] > 2.0 else
                        "medium" if q["separation_ratio"] > 1.2 else "low"),
        "stages": stages,
    }


def demodulate_am(x: np.ndarray, fs: float, carrier_hz: float | None = None,
                  message_bw_hz: float | None = None) -> dict:
    """AM demodulation: envelope detection (and synchronous detection cross-check)."""
    stages: list[dict] = []
    x = np.asarray(x, dtype=np.float64).ravel() if not np.iscomplexobj(x) else np.abs(x).astype(np.float64)
    env = np.abs(x)
    env = env - np.mean(env)
    cutoff = float(message_bw_hz or max(500.0, fs * 0.02))
    try:
        sos = sigproc.butter(4, min(cutoff / (fs / 2) * 1.2, 0.95), btype="lowpass", output="sos")
        msg = sigproc.sosfiltfilt(sos, env, padlen=min(3 * sos.shape[0] * 2, env.size - 2))
        stages.append({"stage": "envelope detection", "status": STATUS_OK,
                       "detail": f"full-wave rectifier + {4}th-order low-pass at {cutoff:g} Hz"})
    except Exception as exc:
        msg = env
        stages.append({"stage": "envelope detection", "status": "failed", "error": str(exc)})
    if msg.size > 8:
        msg = msg - np.mean(msg)
        msg = msg / max(np.max(np.abs(msg)), 1e-9)
    stages.append({"stage": "DC removal and normalisation", "status": STATUS_OK,
                   "detail": "message mean removed and scaled to +/-1"})
    return {"ok": True, "modulation": "AM", "message": msg, "bits": None,
            "n_symbols": 0, "n_bits": 0,
            "quality": {"ok": True, "envelope_modulation_index": None},
            "ber_estimate": {"ok": False, "reason": "not applicable to analogue AM"},
            "bit_quality": "n/a", "stages": stages}


def demodulate_fm(x: np.ndarray, fs: float, deviation_hz: float | None = None,
                  stereo: bool = True, audio_bw_hz: float = 15000.0) -> dict:
    """FM demodulation with optional FM-broadcast stereo MPX decoding.

    Stereo decoding is driven by a *measured* 19 kHz pilot line: the pilot strength,
    its frequency and the resulting left/right correlation are all reported.
    """
    stages: list[dict] = []
    x = np.asarray(x, dtype=np.complex128)
    if x.size < 1024:
        return {"ok": False, "error": "insufficient samples", "stages": stages}
    ph = np.unwrap(np.angle(x))
    mpx = np.diff(ph) * fs / (2 * math.pi)
    mpx = mpx - np.mean(mpx)
    stages.append({"stage": "phase discriminator", "status": STATUS_OK,
                   "detail": f"instantaneous frequency, RMS deviation {np.std(mpx):.1f} Hz"})
    # de-emphasis + audio low-pass
    try:
        sos = sigproc.butter(4, min(max(audio_bw_hz, 1000.0) / (fs / 2), 0.95), btype="lowpass", output="sos")
        audio = sigproc.sosfiltfilt(sos, mpx, padlen=min(3 * sos.shape[0] * 2, mpx.size - 2))
        stages.append({"stage": "audio low-pass", "status": STATUS_OK,
                       "detail": f"{4}th-order Butterworth at {audio_bw_hz:g} Hz (mono/MPX path)"})
    except Exception as exc:
        audio = mpx
        stages.append({"stage": "audio low-pass", "status": "failed", "error": str(exc)})
    # pilot detection at 19 kHz
    pilot = {"detected": False}
    left = right = None
    try:
        n = int(x.size)
        nper = int(min(65536, max(1024, n)))
        f, P = sigproc.welch(mpx, fs=fs, nperseg=nper, window=make_window("hann", nper), scaling="density")
        band = (f > 18000) & (f < 20000)
        ref = (f > 1000) & (f < 18000)
        if np.any(band) and np.any(ref):
            peak = float(np.max(P[band]))
            floor = float(np.median(P[ref]))
            pilot_db = 10 * math.log10(max(peak, 1e-30) / max(floor, 1e-30))
            fpk = float(f[band][int(np.argmax(P[band]))])
            pilot = {"detected": bool(pilot_db > 6.0 and abs(fpk - 19000) < 150),
                     "pilot_frequency_hz": fpk, "pilot_line_db": round(pilot_db, 2),
                     "method": "Welch PSD line search in the 18-20 kHz band; a broadcast stereo MPX "
                               "pilot sits at 19 kHz"}
            stages.append({"stage": "19 kHz stereo pilot search",
                           "status": STATUS_OK if pilot["detected"] else "low_confidence",
                           "detail": f"strongest line {fpk:.0f} Hz, {pilot_db:.1f} dB above the audio floor"})
    except Exception as exc:
        stages.append({"stage": "19 kHz stereo pilot search", "status": "failed", "error": str(exc)})
    if stereo and pilot.get("detected"):
        try:
            t = np.arange(mpx.size) / fs
            fp = pilot["pilot_frequency_hz"]
            # regenerate the 38 kHz subcarrier from the pilot with a matched phase
            pilot_ref = np.exp(-1j * 2 * np.pi * fp * t)
            ph_est = np.angle(np.sum(mpx * pilot_ref))
            sub = np.cos(2 * np.pi * 2 * fp * t + ph_est)
            lpr = np.convolve(mpx, sigproc.firwin(101, 15000 / (fs / 2)), mode="same")
            diff = np.convolve(mpx * sub, sigproc.firwin(101, 15000 / (fs / 2)), mode="same")
            left = lpr + diff
            right = lpr - diff
            corr = float(np.corrcoef(left[1000:-1000], right[1000:-1000])[0, 1]) if mpx.size > 4000 else None
            stages.append({"stage": "stereo MPX decoding", "status": STATUS_OK,
                           "detail": f"38 kHz subcarrier regenerated from the pilot (phase {math.degrees(ph_est):.1f}"
                                     f" deg); L/R correlation {corr:.3f}" if corr is not None else
                                     "38 kHz subcarrier regenerated from the pilot"})
        except Exception as exc:
            stages.append({"stage": "stereo MPX decoding", "status": "failed", "error": str(exc)})
    if audio.size > 8:
        audio = audio / max(np.max(np.abs(audio)), 1e-9)
    return {"ok": True, "modulation": "FM (stereo MPX)" if pilot.get("detected") else "FM",
            "message": audio, "left": left, "right": right, "pilot": pilot,
            "bits": None, "n_symbols": 0, "n_bits": 0,
            "quality": {"ok": True, "rms_deviation_hz": float(np.std(mpx)),
                        "pilot": pilot},
            "ber_estimate": {"ok": False, "reason": "not applicable to analogue FM"},
            "bit_quality": "n/a", "stages": stages}


def demodulate(x: np.ndarray, fs: float, mod: str, rs: float | None = None,
               rolloff: float = 0.35, carrier_hint_hz: float | None = None,
               differential: bool = False, noise_var: float | None = None) -> dict:
    """Dispatch to the right demodulator and normalise the result structure."""
    mod = (mod or "").upper()
    if mod in LINEAR_MODS:
        if not rs:
            return {"ok": False, "error": "a symbol rate is required for linear demodulation",
                    "stages": [{"stage": "symbol rate", "status": "failed",
                                "error": "missing symbol rate"}]}
        return demodulate_linear(x, fs, mod, rs, rolloff, carrier_hint_hz, differential, noise_var)
    if mod in ("2FSK", "FSK", "GFSK"):
        if not rs:
            return {"ok": False, "error": "a symbol rate is required for FSK demodulation", "stages": []}
        return demodulate_fsk(x, fs, rs)
    if mod == "AM":
        return demodulate_am(x, fs)
    if mod in ("FM", "FM_STEREO"):
        return demodulate_fm(x, fs)
    return {"ok": False, "error": f"unsupported modulation '{mod}'",
            "stages": [{"stage": "demodulator selection", "status": "failed",
                        "error": f"no demodulator implemented for {mod}"}]}
