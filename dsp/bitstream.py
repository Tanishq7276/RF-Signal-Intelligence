"""Bit / byte level analysis utilities.

Everything here is deterministic computation on the recovered bit array:
bit<->byte packing, entropy, byte distribution, run-length statistics, repeated
pattern discovery and *candidate* frame-boundary detection.  Nothing is labelled
as "decoded text": only "printable ASCII candidate" as required by SIH §17.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
#  Packing / unpacking
# --------------------------------------------------------------------------- #
def bits_to_bytes(bits: Sequence[int] | np.ndarray, msb_first: bool = True) -> np.ndarray:
    """Pack a bit sequence into bytes (partial trailing byte is zero-padded)."""
    b = np.asarray(bits, dtype=np.uint8).ravel()
    if b.size == 0:
        return np.zeros(0, dtype=np.uint8)
    n_pad = (-b.size) % 8
    if n_pad:
        b = np.concatenate([b, np.zeros(n_pad, dtype=np.uint8)])
    b = b.reshape(-1, 8)
    weights = (128, 64, 32, 16, 8, 4, 2, 1) if msb_first else (1, 2, 4, 8, 16, 32, 64, 128)
    return (b * np.array(weights, dtype=np.uint16)).sum(axis=1).astype(np.uint8)


def bytes_to_bits(data: bytes | bytearray | np.ndarray, msb_first: bool = True) -> np.ndarray:
    arr = np.frombuffer(bytes(data), dtype=np.uint8) if not isinstance(data, np.ndarray) else data.astype(np.uint8)
    if arr.size == 0:
        return np.zeros(0, dtype=np.uint8)
    bits = np.unpackbits(arr)
    if msb_first:
        return bits
    return bits.reshape(-1, 8)[:, ::-1].ravel()


def bytes_to_bit_list(data: np.ndarray, msb_first: bool = True) -> np.ndarray:
    return bytes_to_bits(data, msb_first)


def hex_dump(data: np.ndarray, max_bytes: int = 512, width: int = 16) -> list[str]:
    """Classic hex+ASCII dump lines."""
    d = np.asarray(data, dtype=np.uint8).ravel()[:max_bytes]
    out = []
    for off in range(0, d.size, width):
        chunk = d[off:off + width]
        hx = " ".join(f"{v:02x}" for v in chunk)
        asc = "".join(chr(v) if 32 <= v < 127 else "." for v in chunk)
        out.append(f"{off:08x}  {hx:<{width*3}}  |{asc}|")
    return out


# --------------------------------------------------------------------------- #
#  Statistics
# --------------------------------------------------------------------------- #
def shannon_entropy_bytes(data: np.ndarray) -> float:
    d = np.asarray(data, dtype=np.uint8).ravel()
    if d.size == 0:
        return 0.0
    counts = np.bincount(d, minlength=256).astype(np.float64)
    p = counts[counts > 0] / d.size
    return float(-np.sum(p * np.log2(p)))


def shannon_entropy_bits(bits: np.ndarray) -> float:
    b = np.asarray(bits, dtype=np.uint8).ravel()
    if b.size == 0:
        return 0.0
    p1 = float(b.mean())
    if p1 <= 0.0 or p1 >= 1.0:
        return 0.0
    return float(-(p1 * math.log2(p1) + (1 - p1) * math.log2(1 - p1)))


def byte_histogram(data: np.ndarray) -> dict[str, list[int]]:
    d = np.asarray(data, dtype=np.uint8).ravel()
    return {"counts": np.bincount(d, minlength=256).tolist(), "total": int(d.size)}


def run_length_stats(bits: np.ndarray) -> dict[str, Any]:
    b = np.asarray(bits, dtype=np.uint8).ravel()
    if b.size < 2:
        return {"max_run": 0, "mean_run": 0.0, "runs_gt_8": 0}
    change = np.flatnonzero(np.diff(b.astype(np.int8)) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [b.size]])
    runs = ends - starts
    return {
        "max_run": int(runs.max()),
        "mean_run": float(runs.mean()),
        "runs_gt_8": int(np.sum(runs > 8)),
        "n_runs": int(runs.size),
    }


def bit_autocorrelation(bits: np.ndarray, max_lag: int = 256) -> list[float]:
    """Normalised autocorrelation of the +/-1 mapped bit stream."""
    b = np.asarray(bits, dtype=np.float64).ravel()
    if b.size < 16:
        return []
    x = 2.0 * b - 1.0
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return []
    max_lag = int(min(max_lag, x.size - 1))
    ac = []
    for lag in range(1, max_lag + 1):
        v = float(np.dot(x[:-lag], x[lag:]) / denom)
        ac.append(round(v, 5))
    return ac


def repeat_periodicity(bits: np.ndarray, max_period: int = 4096) -> dict[str, Any]:
    """Detect dominant repetition period using the autocorrelation peak."""
    ac = bit_autocorrelation(bits, max_lag=min(max_period, max(32, int(np.asarray(bits).size // 4))))
    if not ac:
        return {"period_bits": None, "score": 0.0}
    arr = np.asarray(ac, dtype=np.float64)
    best = int(np.argmax(arr))
    score = float(arr[best])
    if score < 0.15:
        return {"period_bits": None, "score": round(score, 4)}
    return {"period_bits": best + 1, "score": round(score, 4)}


def repeated_ngrams(data: np.ndarray, n: int = 4, min_count: int = 3, max_report: int = 12) -> list[dict]:
    """Find n-byte sequences that repeat, with their occurrence spacing."""
    d = np.asarray(data, dtype=np.uint8).ravel()
    if d.size < n * min_count:
        return []
    keys: dict[bytes, list[int]] = {}
    for i in range(0, d.size - n + 1):
        key = d[i:i + n].tobytes()
        keys.setdefault(key, []).append(i)
    out = []
    for key, pos in keys.items():
        if len(pos) < min_count:
            continue
        gaps = np.diff(pos)
        out.append({
            "pattern_hex": key.hex(),
            "pattern_ascii": "".join(chr(c) if 32 <= c < 127 else "." for c in key),
            "count": len(pos),
            "first_positions": [int(p) for p in pos[:6]],
            "mean_gap_bytes": float(gaps.mean()) if gaps.size else None,
            "gap_std": float(gaps.std()) if gaps.size else None,
        })
    out.sort(key=lambda r: (-r["count"], r["gap_std"] if r["gap_std"] is not None else 1e9))
    return out[:max_report]


def frame_boundary_candidates(data: np.ndarray, min_period: int = 2, max_period: int = 2048,
                              min_occurrences: int = 4, max_report: int = 6) -> list[dict]:
    """Auto header candidate detection.

    Scans for short byte patterns that recur at a *regular* period, which is the
    signature of a framed protocol (sync word + payload).  Each candidate is
    scored by (occurrences, regularity of spacing).  These are hypotheses, never
    declared as protocol identifications.
    """
    d = np.asarray(data, dtype=np.uint8).ravel()
    cands: list[dict] = []
    for n in (2, 3, 4, 6, 8):
        if d.size < n * min_occurrences:
            continue
        for rec in repeated_ngrams(d, n=n, min_count=min_occurrences, max_report=8):
            gaps = np.diff(np.array(rec["first_positions"][:min(6, rec["count"])], dtype=int))
            if rec["count"] < min_occurrences:
                continue
            if rec["mean_gap_bytes"] is None:
                continue
            period = rec["mean_gap_bytes"]
            if not (min_period <= period <= max_period):
                continue
            std = rec["gap_std"] or 0.0
            regularity = 1.0 / (1.0 + std / max(period, 1.0))
            score = min(1.0, (rec["count"] / 12.0)) * regularity
            cands.append({
                "pattern_hex": rec["pattern_hex"],
                "pattern_ascii": rec["pattern_ascii"],
                "pattern_bytes": n,
                "occurrences": rec["count"],
                "period_bytes": round(period, 3),
                "period_jitter": round(std, 3),
                "regularity": round(regularity, 4),
                "score": round(score, 4),
                "label": "frame/sync candidate (hypothesis)",
            })
    # keep the best candidate per pattern length
    best: dict[int, dict] = {}
    for c in sorted(cands, key=lambda r: -r["score"]):
        best.setdefault(c["pattern_bytes"], c)
    out = sorted(best.values(), key=lambda r: -r["score"])[:max_report]
    return out


def printable_ascii_candidate(data: np.ndarray, window: int = 16, min_ratio: float = 0.85,
                             max_report: int = 12) -> list[dict]:
    """Locate windows that *look* printable.  Strictly labelled as a candidate."""
    d = np.asarray(data, dtype=np.uint8).ravel()
    if d.size < window:
        return []
    printable = ((d >= 32) & (d < 127)).astype(np.float64)
    csum = np.concatenate([[0.0], np.cumsum(printable)])
    ratios = (csum[window:] - csum[:-window]) / window
    hits = np.flatnonzero(ratios >= min_ratio)
    if hits.size == 0:
        return []
    groups = np.split(hits, np.flatnonzero(np.diff(hits) > window))
    out = []
    for g in groups[-max_report:]:
        start = int(g[0])
        end = int(g[-1]) + window
        seg = d[start:end]
        text = "".join(chr(c) if 32 <= c < 127 else "." for c in seg[:160])
        out.append({
            "start_byte": start,
            "length_bytes": int(end - start),
            "printable_ratio": round(float(printable[start:end].mean()), 4),
            "text_preview": text,
            "label": "printable ASCII candidate",
        })
    return out


def summarise_bits(bits: np.ndarray, n_max: int = 4096) -> dict[str, Any]:
    b = np.asarray(bits, dtype=np.uint8).ravel()
    return {
        "ok": bool(b.size > 0),
        "n_bits": int(b.size),
        "ones_fraction": round(float(b.mean()), 5) if b.size else None,
        "bit_entropy_per_bit": round(shannon_entropy_bits(b), 4),
        "run_length": run_length_stats(b),
        "bit_preview": "".join(str(int(v)) for v in b[:min(256, b.size)]),
        "truncated_preview": b.size > 256,
    }
