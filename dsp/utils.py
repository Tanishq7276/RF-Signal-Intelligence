"""Shared numeric helpers, confidence containers and plotting-downsample utilities.

Design rule (SIH requirement §28/§21): every automatic result in this platform is
represented as a dict carrying an explicit *status*, *confidence*, *method* and
*limitations*.  Nothing is ever returned as a bare number without provenance.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
#  Confidence / evidence containers
# --------------------------------------------------------------------------- #
STATUS_OK = "ok"
STATUS_LOW = "low_confidence"
STATUS_UNAVAILABLE = "unable"          # "Unable to estimate reliably"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"


def clamp01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def pct(conf: float) -> float:
    """Confidence in percent, rounded."""
    return round(100.0 * clamp01(conf), 1)


def param(
    name: str,
    value: Any,
    unit: str | None = None,
    confidence: float | None = None,
    method: str = "",
    status: str = STATUS_OK,
    evidence: Sequence[str] | None = None,
    limitations: Sequence[str] | None = None,
    note: str | None = None,
) -> dict:
    """Build a provenance-carrying parameter record (value + confidence + method)."""
    rec: dict = {
        "name": name,
        "value": value,
        "unit": unit,
        "status": status,
        "confidence": None if confidence is None else round(100.0 * clamp01(confidence), 1),
        "method": method or "not specified",
        "evidence": list(evidence or []),
        "limitations": list(limitations or []),
    }
    if status == STATUS_UNAVAILABLE:
        rec["value"] = None
        rec["display"] = "Unable to estimate reliably"
    if note:
        rec["note"] = note
    return rec


def unavailable(name: str, reason: str, method: str = "not applicable", unit: str | None = None) -> dict:
    return param(name, None, unit, None, method, STATUS_UNAVAILABLE, limitations=[reason])


def human_si(v: float | None, unit: str = "", digits: int = 4) -> str:
    """Format 1234567.0 -> '1.235 MHz' (display helper used by reports)."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    a = abs(v)
    for scale, prefix in ((1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, "")):
        if a >= scale or scale == 1.0:
            return f"{v/scale:.{digits}g} {prefix}{unit}".strip()
    return f"{v:.{digits}g} {unit}".strip()


def db(x: np.ndarray | float) -> np.ndarray | float:
    """Power -> dB, guarded against log(0)."""
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), 1e-30))


def safe_float(v: Any, default: float | None = None) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


# --------------------------------------------------------------------------- #
#  Array helpers
# --------------------------------------------------------------------------- #
def next_pow2(n: int) -> int:
    return int(2 ** int(math.ceil(math.log2(max(2, n)))))


def decimate_for_plot(x: np.ndarray, max_points: int = 4000) -> np.ndarray:
    """Return an array of at most `max_points` samples for transport to the UI.

    For |x| > max_points the samples are grouped and *both* min and max of every
    group are kept (progressive "envelope downsampling"), so a plot never hides a
    short spike.  The result is therefore length <= 2*max_points.
    """
    x = np.asarray(x)
    n = x.size
    if n <= max_points:
        return x.copy()
    stride = int(math.ceil(n / max_points))
    usable = (n // stride) * stride
    blocks = x[:usable].reshape(-1, stride)
    if np.iscomplexobj(blocks):
        out = np.empty(2 * blocks.shape[0], dtype=x.dtype)
        out[0::2] = blocks[np.arange(blocks.shape[0]), np.argmax(blocks.real, axis=1)]
        out[1::2] = blocks[np.arange(blocks.shape[0]), np.argmin(blocks.real, axis=1)]
        return out
    out = np.empty(2 * blocks.shape[0], dtype=x.dtype)
    out[0::2] = blocks.max(axis=1)
    out[1::2] = blocks.min(axis=1)
    return out


def running_mean(x: np.ndarray, n: int) -> np.ndarray:
    n = max(1, int(n))
    if n == 1:
        return np.asarray(x, dtype=np.float64)
    k = np.ones(n, dtype=np.float64) / n
    return np.convolve(np.asarray(x, dtype=np.float64), k, mode="same")


def robust_sigma(x: np.ndarray) -> float:
    """MAD-based noise sigma (immune to outliers / carriers)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    return float(1.4826 * np.median(np.abs(x - med)))


def parabolic_peak(y: np.ndarray, i: int) -> tuple[float, float]:
    """Sub-bin peak refinement.  Returns (refined_index, peak_value)."""
    y = np.asarray(y, dtype=np.float64)
    if y.size < 3:
        return float(i), float(y[i]) if y.size else 0.0
    if i <= 0 or i >= y.size - 1:
        return float(i), float(y[i])
    a, b, c = y[i - 1], y[i], y[i + 1]
    denom = (a - 2 * b + c)
    delta = 0.0 if abs(denom) < 1e-20 else 0.5 * (a - c) / denom
    delta = float(np.clip(delta, -0.5, 0.5))
    return float(i) + delta, float(b - 0.25 * (a - c) * delta)


def welch_style_smooth(y: np.ndarray, width: int = 3) -> np.ndarray:
    if width <= 1:
        return np.asarray(y, dtype=np.float64)
    k = np.hanning(width)
    k /= k.sum()
    return np.convolve(np.asarray(y, dtype=np.float64), k, mode="same")


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays into JSON-safe python objects."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        v = obj.item()
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, complex):
        return {"re": obj.real, "im": obj.imag}
    return obj


def downsample_complex(x: np.ndarray, max_points: int = 3000) -> np.ndarray:
    """Decimate a complex array by block-averaging (used for scatter views)."""
    x = np.asarray(x)
    if x.size <= max_points:
        return x
    stride = int(np.ceil(x.size / max_points))
    usable = (x.size // stride) * stride
    return x[:usable].reshape(-1, stride).mean(axis=1)


def occupied_bandwidth(freqs: np.ndarray, psd: np.ndarray, fraction: float = 0.99) -> tuple[float, float, float]:
    """Occupied bandwidth containing `fraction` of total power.

    Returns (f_lo, f_hi, bandwidth).  Pure integration of the measured PSD.
    """
    p = np.asarray(psd, dtype=np.float64).ravel()
    f = np.asarray(freqs, dtype=np.float64).ravel()
    if p.size < 4 or np.all(p <= 0):
        return float(f.min()), float(f.max()), float(f.max() - f.min())
    total = p.sum()
    c = np.cumsum(p) / total
    lo_frac = (1.0 - fraction) / 2.0
    hi_frac = 1.0 - lo_frac
    i_lo = int(np.searchsorted(c, lo_frac))
    i_hi = int(np.searchsorted(c, hi_frac))
    i_lo = int(np.clip(i_lo, 0, f.size - 1))
    i_hi = int(np.clip(i_hi, 0, f.size - 1))
    return float(f[i_lo]), float(f[i_hi]), float(f[i_hi] - f[i_lo])


def estimate_noise_floor_db(psd_db: np.ndarray, percentile: float = 25.0) -> float:
    """Robust spectral noise-floor estimate in dB (low percentile of the PSD)."""
    p = np.asarray(psd_db, dtype=np.float64).ravel()
    p = p[np.isfinite(p)]
    if p.size == 0:
        return float("nan")
    return float(np.percentile(p, percentile))


def freq_offset_power_weighted(freqs: np.ndarray, psd: np.ndarray, flo: float, fhi: float) -> float:
    m = (freqs >= flo) & (freqs <= fhi)
    if not np.any(m):
        return float("nan")
    p = psd[m]
    return float(np.sum(freqs[m] * p) / max(np.sum(p), 1e-30))


def compact(obj: Any, max_list: int = 64, max_str: int = 400) -> Any:
    """Recursively trim containers so analysis payloads stay small on the wire.

    Long lists (per-sample arrays that leaked into a stage detail) are cut to ``max_list`` items and
    a marker is appended so nothing looks silently truncated.
    """
    if isinstance(obj, dict):
        return {k: compact(v, max_list, max_str) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        out = [compact(v, max_list, max_str) for v in obj[:max_list]]
        if len(obj) > max_list:
            out.append(f"... {len(obj) - max_list} more items omitted")
        return out
    if isinstance(obj, str):
        return obj if len(obj) <= max_str else obj[:max_str] + "..."
    if isinstance(obj, np.ndarray):
        return compact(obj.tolist(), max_list, max_str)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj
