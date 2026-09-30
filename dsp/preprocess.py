"""Signal preprocessing: DC removal, normalisation, detrending, filtering,
resampling, spectral-subtraction denoising and windowing.

Every operation reports what was actually done (filter order, cut-offs, scale
factors) plus before/after measurements so the UI can show a verifiable
"Original -> Processed" comparison rather than a decorative animation.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import signal as sigproc

from .utils import param, STATUS_OK, STATUS_LOW, STATUS_UNAVAILABLE, db

# --------------------------------------------------------------------------- #
#  Windows
# --------------------------------------------------------------------------- #
from scipy.signal import windows as _win

WINDOWS = {
    "hann": lambda n: _win.hann(n, sym=False),
    "hamming": lambda n: _win.hamming(n, sym=False),
    "blackman": lambda n: _win.blackman(n, sym=False),
    "blackmanharris": lambda n: _win.blackmanharris(n, sym=False),
    "flattop": lambda n: _win.flattop(n, sym=False),
    "rectangular": lambda n: np.ones(n),
    "kaiser": lambda n: _win.kaiser(n, 8.6, sym=False),
    "tukey": lambda n: _win.tukey(n, 0.25, sym=False),
}


def make_window(name: str, n: int) -> np.ndarray:
    fn = WINDOWS.get((name or "hann").lower(), np.hanning)
    w = fn(max(2, int(n)))
    return np.asarray(w, dtype=np.float64)


def window_coherent_gain(name: str, n: int = 1024) -> float:
    """Coherent gain of the window (used to correct PSD amplitude)."""
    return float(np.mean(make_window(name, n)))


def window_enbw(name: str, n: int = 1024) -> float:
    """Equivalent noise bandwidth in bins (PSD scaling)."""
    w = make_window(name, n)
    return float(n * np.sum(w ** 2) / max(np.sum(w), 1e-20) ** 2 * (np.sum(w) / n) / (np.sum(w) / n))


# --------------------------------------------------------------------------- #
#  Basic operations
# --------------------------------------------------------------------------- #
def remove_dc(x: np.ndarray) -> tuple[np.ndarray, dict]:
    m = complex(np.mean(x)) if np.iscomplexobj(x) else float(np.mean(x))
    y = x - m
    return y, {"dc_removed": {"real": m.real, "imag": m.imag if np.iscomplexobj(x) else 0.0},
               "dc_magnitude": float(abs(m))}


def normalize(x: np.ndarray, mode: str = "rms") -> tuple[np.ndarray, dict]:
    if mode == "peak":
        g = float(np.max(np.abs(x)))
        kind = "peak"
    elif mode == "max_abs":
        g = float(np.max(np.abs(x)))
        kind = "peak (max |sample|)"
    elif mode == "none":
        return x.copy(), {"normalisation": "none"}
    else:
        g = float(np.sqrt(np.mean(np.abs(x) ** 2)))
        kind = "RMS"
    if g <= 0 or not math.isfinite(g):
        return x.copy(), {"normalisation": "skipped (zero or non-finite power)",
                          "status": STATUS_UNAVAILABLE}
    return x / g, {"normalisation": kind, "gain_applied": 1.0 / g}


def detrend(x: np.ndarray, mode: str = "linear", cutoff_hz: float | None = None,
            fs: float | None = None) -> tuple[np.ndarray, dict]:
    """Remove trend: 'linear' (least-squares ramp), 'mean', or 'moving_average'."""
    if mode == "none":
        return x.copy(), {"detrend": "none"}
    if mode == "linear":
        n = x.size
        t = np.arange(n, dtype=np.float64)
        A = np.column_stack([t, np.ones(n)])
        if np.iscomplexobj(x):
            c, *_ = np.linalg.lstsq(A, x, rcond=None)
        else:
            c, *_ = np.linalg.lstsq(A, x.astype(np.float64), rcond=None)
        trend = A @ c
        return x - trend, {"detrend": "linear least-squares ramp removed",
                           "slope_per_sample": {"re": float(np.real(c[0])), "im": float(np.imag(c[0]))}}
    if mode == "moving_average":
        w = int(cutoff_hz or 128)
        w = max(3, min(w, max(3, x.size // 4)))
        k = np.ones(w, dtype=np.float64) / w
        trend = np.convolve(x, k, mode="same")
        if x.size > 8 and w > 8:
            edge = w // 2
            trend[:edge] = trend[edge]
            trend[-edge:] = trend[-edge - 1]
        return x - trend, {"detrend": f"moving-average baseline (window {w} samples) removed"}
    m = np.mean(x)
    return x - m, {"detrend": "mean removed"}


def design_filter(kind: str, fs: float, band: list[float] | None = None, order: int = 6,
                  ripple: float = 60.0) -> dict:
    """Design a Butterworth (or Chebyshev-II for steep specs) digital filter."""
    nyq = 0.5 * fs
    order = int(np.clip(order, 2, 16))
    if kind not in ("lowpass", "highpass", "bandpass", "bandstop"):
        raise ValueError(f"unsupported filter kind '{kind}'")
    band = list(band or [])
    if kind in ("lowpass", "highpass"):
        if not band:
            raise ValueError("cutoff frequency required")
        f_c = float(band[0])
        if not (0 < f_c < nyq):
            raise ValueError(f"cutoff {f_c:g} Hz outside (0, {nyq:g}) for fs={fs:g}")
        wn = f_c / nyq
        sos = sigproc.butter(order, wn, btype=kind, output="sos")
        desc = f"{order}th-order Butterworth {kind} @ {f_c:g} Hz"
    else:
        if len(band) < 2:
            raise ValueError("two band edges required")
        lo, hi = sorted([float(band[0]), float(band[1])])
        if not (0 < lo < hi < nyq):
            raise ValueError(f"band {lo:g}-{hi:g} Hz invalid for fs={fs:g}")
        sos = sigproc.butter(order, [lo / nyq, hi / nyq], btype=kind, output="sos")
        desc = f"{order}th-order Butterworth {kind} {lo:g}-{hi:g} Hz"
    return {"sos": sos, "description": desc, "kind": kind, "order": order,
            "band": band, "fs": fs}


def apply_filter(x: np.ndarray, flt: dict, zero_phase: bool = True) -> tuple[np.ndarray, dict]:
    sos = flt["sos"]
    padlen = min(3 * (sos.shape[0] * 2), max(0, x.size - 1))
    if zero_phase and x.size > padlen + 2:
        y = sigproc.sosfiltfilt(sos, x.astype(np.complex128 if np.iscomplexobj(x) else np.float64),
                                padlen=padlen)
        mode = "zero-phase (forward-backward, filtfilt)"
    else:
        y = sigproc.sosfilt(sos, x.astype(np.complex128 if np.iscomplexobj(x) else np.float64))
        mode = "causal single-pass (sosfilt)"
    return y, {"filter": flt["description"], "filter_mode": mode,
               "group_delay_compensated": bool(zero_phase)}


def correct_iq_imbalance(x: np.ndarray, mode: str = "blind_moments") -> tuple[np.ndarray, dict]:
    """Blind I/Q imbalance correction (Gram-Schmidt on second-order moments).

    Estimates the gain ratio g and phase error phi between the I and Q streams
    from the covariance of the baseband components and applies the standard
    correction.  Also reports the image-rejection ratio before and after, which is
    a *measured* quantity - not an estimate of intent.
    """
    x = np.asarray(x)
    if not np.iscomplexobj(x):
        return x.copy(), {"iq_correction": "not applicable (real-valued signal)"}
    i = x.real.astype(np.float64)
    q = x.imag.astype(np.float64)
    ei, eq = float(np.mean(i ** 2)), float(np.mean(q ** 2))
    if ei <= 0 or eq <= 0:
        return x.copy(), {"iq_correction": "skipped (zero-power I or Q stream)"}
    rho = float(np.mean(i * q)) / math.sqrt(ei * eq)
    gain = math.sqrt(eq / ei)
    # phase error from the normalised cross-moment
    phi = math.asin(max(-1.0, min(1.0, rho)))
    qc = (q - math.tan(phi) * i) / (math.cos(phi) * gain) if abs(math.cos(phi)) > 1e-9 else q
    y = i + 1j * qc
    y = y * math.sqrt(float(np.mean(np.abs(x) ** 2)) / max(float(np.mean(np.abs(y) ** 2)), 1e-30))

    def irr(sig: np.ndarray) -> float | None:
        """Spectral-asymmetry (image proxy) in dB: upper vs mirrored lower sideband."""
        n = int(min(8192, sig.size))
        if n < 64:
            return None
        seg = np.asarray(sig[:n], dtype=np.complex128) * np.hanning(n)
        P = np.abs(np.fft.fftshift(np.fft.fft(seg))) ** 2
        half = P.size // 2
        pos = float(np.sum(P[half + 1:]))
        neg = float(np.sum(P[1:half][::-1]))
        if neg <= 0:
            return None
        return 10 * math.log10(max(pos, 1e-30) / neg)

    return y, {"iq_correction": "blind Gram-Schmidt (second-order moments)",
               "gain_imbalance_db": round(20 * math.log10(max(gain, 1e-9)), 3),
               "phase_imbalance_deg": round(math.degrees(phi), 3),
               "iq_correlation_before": round(rho, 4),
               "spectral_asymmetry_db_before": round(irr(x), 3) if irr(x) is not None else None,
               "spectral_asymmetry_db_after": round(irr(y), 3) if irr(y) is not None else None,
               "note": "spectral asymmetry is a proxy for image content; it is only meaningful for "
                       "signals whose true spectrum is symmetric about the carrier"}


def analytic_signal(x: np.ndarray) -> np.ndarray:
    return sigproc.hilbert(np.asarray(x, dtype=np.float64))


def resample(x: np.ndarray, fs: float, new_fs: float) -> tuple[np.ndarray, float, dict]:
    """Rational polyphase resampling with an explicit alias-risk warning."""
    from math import gcd
    if new_fs <= 0 or fs <= 0:
        raise ValueError("sample rates must be positive")
    g = gcd(int(round(fs)), int(round(new_fs)))
    up, down = int(round(new_fs)) // g, int(round(fs)) // g
    if up > 4096 or down > 4096:
        # fall back to a simpler ratio approximation
        from fractions import Fraction
        fr = Fraction(int(round(new_fs)), int(round(fs))).limit_denominator(2048)
        up, down = fr.numerator, fr.denominator
    y = sigproc.resample_poly(x, up, down, window=("kaiser", 8.6))
    info = {
        "up": up, "down": down, "ratio": round(new_fs / fs, 8),
        "method": f"polyphase rational resampling ({up}/{down}) with Kaiser-windowed anti-alias FIR",
    }
    if new_fs < fs:
        info["note"] = (f"decreasing the rate to {new_fs:g} Hz keeps content only below "
                        f"{new_fs/2:g} Hz; any signal above that would alias")
    return y, float(new_fs), info


def spectral_subtract(x: np.ndarray, fs: float, noise_percentile: float = 20.0,
                      nperseg: int = 1024, over_subtraction: float = 1.0,
                      floor_db: float = -18.0) -> tuple[np.ndarray, dict]:
    """STFT spectral-subtraction denoiser (real operation, honest bookkeeping).

    Estimates a per-bin noise magnitude from the quietest `noise_percentile`% of
    frames and attenuates each bin accordingly (Wiener-style gain with a floor).
    """
    x = np.asarray(x)
    nperseg = int(min(nperseg, max(64, x.size // 4)))
    noverlap = nperseg // 2
    f, t, Z = sigproc.stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap,
                           window="hann", return_onesided=False)
    mag = np.abs(Z)
    if mag.size == 0:
        return x.copy(), {"denoise": "skipped (STFT too short)"}
    noise = np.percentile(mag, noise_percentile, axis=1, keepdims=True)
    noise = np.maximum(noise, 1e-12)
    gain = np.maximum(1.0 - over_subtraction * (noise / np.maximum(mag, 1e-12)), 10 ** (floor_db / 20.0))
    Zc = Z * gain
    _, y = sigproc.istft(Zc, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hann",
                         input_onesided=False)
    y = y[: x.size]
    if y.size < x.size:
        y = np.pad(y, (0, x.size - y.size))
    removed = float(np.mean(np.abs(x) ** 2) - np.mean(np.abs(y) ** 2))
    return y, {"denoise": f"STFT spectral subtraction (nperseg={nperseg}, noise= {noise_percentile}th pct per bin, "
                          f"gain floor {floor_db} dB)",
               "noise_estimation": "per-frequency-bin percentile over time",
               "power_removed_fraction": round(max(0.0, removed / max(np.mean(np.abs(x) ** 2), 1e-30)), 4)}


# --------------------------------------------------------------------------- #
#  Measurements (before / after)
# --------------------------------------------------------------------------- #
def signal_metrics(x: np.ndarray, fs: float | None = None) -> dict:
    """Amplitude/phase statistics of a complex (or real) sample block."""
    x = np.asarray(x)
    n = x.size
    if n == 0:
        return {"n_samples": 0}
    is_c = np.iscomplexobj(x)
    mag = np.abs(x).astype(np.float64)
    power = mag ** 2
    mean_p = float(np.mean(power))
    rms = math.sqrt(mean_p)
    peak = float(np.max(mag))
    m: dict[str, Any] = {
        "n_samples": int(n),
        "rms": rms,
        "peak": peak,
        "crest_factor_db": round(20 * math.log10(peak / rms), 3) if rms > 0 else None,
        "power_dbfs": round(10 * math.log10(mean_p), 3) if mean_p > 0 else None,
        "magnitude_mean": float(np.mean(mag)),
        "magnitude_std": float(np.std(mag)),
        "magnitude_median": float(np.median(mag)),
        "dc_real": float(np.mean(x.real)),
    }
    if is_c:
        m["dc_imag"] = float(np.mean(x.imag))
        m["dc_magnitude"] = float(abs(complex(m["dc_real"], m["dc_imag"])))
        m["iq_power_ratio_db"] = round(10 * math.log10(
            float(np.mean(x.real ** 2)) / max(float(np.mean(x.imag ** 2)), 1e-30)), 3)
        ph = np.unwrap(np.angle(x))
        m["phase_std_rad"] = float(np.std(np.angle(x)))
        m["phase_unwrapped_std_rad"] = float(np.std(ph - np.polyval(np.polyfit(
            np.arange(n, dtype=np.float64), ph, 1)[0:2] if n > 8 else [0, 0], np.arange(n))))
    else:
        m["envelope_following"] = "real-valued signal: quadrature/phase metrics not applicable"
    if fs:
        m["duration_s"] = n / fs
    return m


def compare_metrics(before: dict, after: dict) -> list[dict]:
    """Delta table for the Original -> Processed view."""
    keys = ["rms", "peak", "crest_factor_db", "power_dbfs", "dc_magnitude", "iq_power_ratio_db",
            "magnitude_mean", "phase_std_rad"]
    out = []
    for k in keys:
        b, a = before.get(k), after.get(k)
        if b is None or a is None:
            continue
        out.append({"metric": k, "before": b, "after": a, "delta": a - b})
    return out


# --------------------------------------------------------------------------- #
#  Pipeline
# --------------------------------------------------------------------------- #
DEFAULT_STEPS = {
    "remove_dc": True,
    "detrend": "none",
    "normalize": "none",
    "filter": "none",          # none | lowpass | highpass | bandpass | bandstop
    "filter_band": [],
    "filter_order": 6,
    "denoise": False,
    "correct_iq": False,
    "denoise_strength": 20.0,
    "resample_to": None,
    "window": "hann",
}


def run_pipeline(x: np.ndarray, fs: float | None, steps: dict | None = None) -> dict:
    """Apply the configured preprocessing chain, recording every operation."""
    cfg = dict(DEFAULT_STEPS)
    cfg.update(steps or {})
    log: list[dict] = []
    y = np.asarray(x).copy()
    fs_out = fs
    before = signal_metrics(y, fs)

    def _step(name: str, fn):
        nonlocal y, fs_out
        try:
            y, info = fn()
            log.append({"step": name, "status": STATUS_OK, **info})
        except Exception as exc:
            log.append({"step": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})

    if cfg.get("remove_dc"):
        _step("DC removal", lambda: remove_dc(y))
    if cfg.get("detrend", "none") != "none":
        _step("Detrending", lambda: detrend(y, cfg["detrend"],
                                           cfg.get("detrend_window"), fs))
    if cfg.get("filter", "none") != "none":
        def _filt():
            flt = design_filter(cfg["filter"], fs or 1.0, cfg.get("filter_band"), cfg.get("filter_order", 6))
            yy, info = apply_filter(y, flt)
            return yy, info
        _step(f"{cfg['filter']} filter", _filt)
    if cfg.get("correct_iq") and np.iscomplexobj(y):
        _step("I/Q imbalance correction", lambda: correct_iq_imbalance(y))
    if cfg.get("denoise"):
        _step("Noise reduction", lambda: spectral_subtract(y, fs or 1.0,
                                                           cfg.get("denoise_strength", 20.0)))
    if cfg.get("normalize", "none") != "none":
        _step(f"Normalisation ({cfg['normalize']})", lambda: normalize(y, cfg["normalize"]))
    if cfg.get("resample_to"):
        def _rs():
            nonlocal fs_out
            yy, fs_new, info = resample(y, fs or 1.0, float(cfg["resample_to"]))
            fs_out = fs_new
            return yy, info
        _step("Resampling", _rs)

    after = signal_metrics(y, fs_out)
    return {
        "samples": y,
        "fs": fs_out,
        "log": log,
        "before": before,
        "after": after,
        "deltas": compare_metrics(before, after),
        "steps_applied": [s["step"] for s in log if s.get("status") == STATUS_OK],
        "failed_steps": [s for s in log if s.get("status") != STATUS_OK],
        "window": cfg.get("window", "hann"),
        "alias_warning": next((s.get("note") for s in log if s.get("step") == "Resampling"), None),
    }
