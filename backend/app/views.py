"""Trimming of analysis payloads for the browser.

The cache file keeps *everything* (including the demodulated bits, LLRs and symbol samples) so the
module endpoints can reuse the exact stream that was analysed.  The browser does not need those
arrays: they are dropped (or capped) here, with an explicit marker instead of a silent truncation,
and float arrays are rounded so the JSON stays small.
"""
from __future__ import annotations

import math
from typing import Any

#: keys that are large per-sample/per-symbol arrays and are only sent on request
HEAVY_KEYS = ("bits", "llrs", "soft", "symbols", "traces_i", "traces_q", "symbol_i", "symbol_q",
              "i_full", "q_full")

MARK = "... {n} more items omitted (request the dedicated endpoint for the full array)"


def _round(value: float) -> float:
    if not math.isfinite(value):
        return None                                                # type: ignore[return-value]
    if value == 0.0:
        return 0.0
    if abs(value) >= 1e5 or abs(value) < 1e-4:
        return round(value, 9)
    return round(value, 6)


def trim(obj: Any, max_list: int = 4000, drop_heavy: bool = True,
         skip_keys: tuple[str, ...] = (), drop_keys: tuple[str, ...] = ("_streams",)) -> Any:
    if isinstance(obj, dict):
        out: dict = {}
        for k, v in obj.items():
            if k in drop_keys:
                continue
            if k in skip_keys:
                out[k] = f"omitted from this payload ({type(v).__name__})"
            elif drop_heavy and k in HEAVY_KEYS and isinstance(v, (list, tuple)) and len(v) > 64:
                out[k] = {"_omitted": True, "n": len(v), "note": MARK.format(n=len(v)),
                          "preview": list(v[:16]) if k in ("bits", "bit_preview") else None}
                if out[k]["preview"] is None:
                    del out[k]["preview"]
            else:
                out[k] = trim(v, max_list, drop_heavy, skip_keys, drop_keys)
        return out
    if isinstance(obj, (list, tuple)):
        n = len(obj)
        out_list = [trim(v, max_list, drop_heavy, skip_keys, drop_keys)
                    for v in obj[:max_list]]
        if n > max_list:
            out_list.append(MARK.format(n=n - max_list))
        return out_list
    if isinstance(obj, float):
        return _round(obj)
    return obj


def api_view(result: dict, spectrogram: bool = False, heavy: bool = False,
             max_list: int = 4000, keep_streams: bool = False) -> dict:
    """Analysis payload for the browser (no bit/LLR arrays, no waterfall matrix by default)."""
    skip = () if spectrogram else ("db",)
    drop = () if keep_streams else ("_streams",)
    out = trim(result, max_list=max_list, drop_heavy=not heavy, skip_keys=skip, drop_keys=drop)
    if isinstance(out, dict) and not spectrogram:
        spec = out.get("spectrogram")
        if isinstance(spec, dict):
            spec.pop("db", None)
            spec["db_omitted"] = ("the waterfall matrix is served by "
                                  "GET /api/analysis/{id}/spectrogram (downsampled and rounded)")
    return out
