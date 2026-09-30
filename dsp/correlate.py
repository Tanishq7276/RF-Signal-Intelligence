"""Known-pattern correlation search and automatic repeated-structure (header) discovery.

Two capabilities:

* **Pattern search** - the user supplies a header as text, hex or a bit string; the platform finds
  every occurrence in the demodulated bitstream, byte-aligned or not, with an exact match or a
  Hamming-distance-tolerant match.  Coarse positions come from an FFT cross-correlation on +/-1
  sequences (so the search is fast even for megabit streams) and every candidate is then verified
  bit by bit, so a reported hit is a *verified* hit.
* **Repeated structure** - the bitstream is folded at every plausible frame length and the
  agreement of each column across frames is measured.  A frame length whose columns agree far more
  than chance reveals a periodic header, and the consensus pattern is reported as a *header
  candidate* with the agreement score, the frame count and the number of ambiguous bits.  It is
  never presented as a decoded protocol header.
"""

from __future__ import annotations

import re

import numpy as np

from .utils import clamp01, human_si, param

__all__ = ["parse_pattern", "bits_to_hex", "find_pattern", "repeated_structure", "auto_discover",
           "analyse_correlation", "correlation_params"]


def bits_to_hex(bits: np.ndarray) -> str:
    b = np.asarray(bits, dtype=np.uint8).ravel()
    n = (b.size // 4) * 4
    if n == 0:
        return ""
    nib = b[:n].reshape(-1, 4) @ (1 << np.arange(3, -1, -1))
    return "".join(f"{v:X}" for v in nib.astype(np.uint8))


def _text_to_bits(text: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(text.encode("latin-1", errors="replace"), dtype=np.uint8)) \
        .astype(np.uint8)


def _hex_to_bits(hex_str: str) -> np.ndarray:
    clean = re.sub(r"[^0-9a-fA-F]", "", hex_str)
    if len(clean) % 2:
        clean = clean[:-1]
    return np.unpackbits(np.frombuffer(bytes.fromhex(clean), dtype=np.uint8)).astype(np.uint8)


def parse_pattern(pattern=None, kind: str = "auto") -> dict:
    """Turn a user-supplied pattern into bits.

    ``kind`` is one of ``auto``, ``text``, ``hex``, ``bits``.  With ``auto`` a string of only
    0/1 and whitespace is read as bits, a string of only hex digits (and an even length) as hex,
    anything else as text.
    """
    if pattern is None:
        return {"ok": False, "message": "no pattern supplied"}
    if isinstance(pattern, (list, tuple, np.ndarray)):
        bits = np.asarray(pattern, dtype=np.uint8).ravel() & 1
        return {"ok": True, "bits": bits, "kind": "bits", "label": f"{bits.size} raw bits",
                "hex": bits_to_hex(bits)}
    s = str(pattern)
    k = (kind or "auto").lower()
    if k == "auto":
        compact = re.sub(r"[\s,;:_-]", "", s)
        if compact and set(compact) <= {"0", "1"} and len(compact) >= 4:
            k = "bits"
        elif compact and set(compact.lower()) <= set("0123456789abcdef") and len(compact) % 2 == 0 \
                and len(compact) >= 4 and any(c.isdigit() for c in compact):
            k = "hex"
        else:
            k = "text"
    if k == "bits":
        compact = re.sub(r"[\s,;:_-]", "", s)
        bits = np.array([1 if c == "1" else 0 for c in compact], dtype=np.uint8)
        return {"ok": bool(bits.size), "bits": bits, "kind": "bits",
                "label": f"{bits.size}-bit pattern", "hex": bits_to_hex(bits)}
    if k == "hex":
        try:
            bits = _hex_to_bits(s)
        except ValueError as exc:
            return {"ok": False, "message": f"invalid hex pattern: {exc}"}
        return {"ok": bool(bits.size), "bits": bits, "kind": "hex",
                "label": f"{bits.size // 8} byte(s) of hex", "hex": bits_to_hex(bits)}
    bits = _text_to_bits(s)
    return {"ok": bool(bits.size), "bits": bits, "kind": "text",
            "label": f"{len(s)} ASCII character(s)", "hex": bits_to_hex(bits)}


def _verify_hits(bits: np.ndarray, needle: np.ndarray, coarse: np.ndarray, max_errors: int,
                 max_hits: int) -> list[dict]:
    hits = []
    L = needle.size
    n = bits.size
    for pos in coarse:
        pos = int(pos)
        if pos < 0 or pos + L > n:
            continue
        if max_errors == 0:
            if np.array_equal(bits[pos:pos + L], needle):
                hits.append({"bit_index": pos, "errors": 0, "score": 1.0, "exact": True})
        else:
            err = int(np.count_nonzero(bits[pos:pos + L] != needle))
            if err <= max_errors:
                hits.append({"bit_index": pos, "errors": err,
                             "score": float(1.0 - err / L), "exact": err == 0})
        if len(hits) >= max_hits:
            break
    # byte-aligned hits are reported separately because they indicate a byte-oriented framing
    for h in hits:
        h["byte_index"] = h["bit_index"] // 8 if h["bit_index"] % 8 == 0 else None
        h["byte_aligned"] = h["bit_index"] % 8 == 0
    return hits


def _coarse_positions(bits: np.ndarray, needle: np.ndarray, min_score: float, limit: int) -> np.ndarray:
    """FFT cross-correlation on +/-1 sequences -> candidate offsets."""
    a = 2.0 * np.asarray(bits, dtype=np.float64) - 1.0
    b = 2.0 * np.asarray(needle, dtype=np.float64) - 1.0
    n = a.size
    L = b.size
    if L > n:
        return np.zeros(0, dtype=int)
    nfft = 1 << int(np.ceil(np.log2(n + L)))
    corr = np.fft.irfft(np.fft.rfft(a, nfft) * np.conj(np.fft.rfft(b, nfft)), nfft)
    # corr[k] = sum_j a[j] b[j-k]: a full-length match at position p peaks at k = p
    valid = corr[: n - L + 1] / L
    thr = min_score
    idx = np.flatnonzero(valid >= thr)
    # keep the strongest candidates, spaced by at least one sample
    order = idx[np.argsort(-valid[idx])]
    keep, taken = [], set()
    for i in order:
        if any(abs(int(i) - j) <= 1 for j in taken):
            continue
        keep.append(int(i))
        taken.add(int(i))
        if len(keep) >= limit:
            break
    return np.array(sorted(keep), dtype=int)


def find_pattern(bits: np.ndarray, needle: np.ndarray, max_errors: int = 0, max_hits: int = 64) -> dict:
    """Search ``bits`` for ``needle`` allowing up to ``max_errors`` bit errors."""
    bits = np.asarray(bits, dtype=np.uint8).ravel() & 1
    needle = np.asarray(needle, dtype=np.uint8).ravel() & 1
    if needle.size == 0 or bits.size < needle.size:
        return {"ok": False, "message": "the pattern is empty or longer than the bitstream",
                "hits": []}
    L = needle.size
    min_score = max(0.0, 1.0 - 2.0 * max_errors / L) if max_errors else 1.0
    coarse = _coarse_positions(bits, needle, min_score * 0.999, limit=max_hits * 6)
    hits = _verify_hits(bits, needle, coarse, max_errors, max_hits)
    return {"ok": True, "hits": hits, "n_hits": len(hits), "needle_bits": int(L),
            "n_bits_searched": int(bits.size), "max_errors": int(max_errors),
            "method": "FFT cross-correlation on +/-1 sequences to shortlist offsets, then a "
                      "bit-by-bit Hamming verification of every candidate",
            "truncated": len(hits) >= max_hits}


def repeated_structure(bits: np.ndarray, min_period: int = 8, max_period: int = 512,
                       min_frames: int = 3, max_candidates: int = 8,
                       alpha: float = 0.01, min_fixed: int = 8,
                       max_bits: int = 200000) -> list[dict]:
    """Fold the bitstream at every plausible frame length and score the resulting structure.

    A column is *fixed* when its majority fraction is significant under the binomial null with a
    Bonferroni-corrected level (or when it reaches 90 % agreement).  Each surviving candidate is
    then validated against its own frame model: ``model_match`` is the fraction of the consensus
    pattern that an average frame actually reproduces, and ``frames_consistent`` the fraction of
    frames that reproduce it almost completely.  A harmonic period (twice the true frame) shows a
    perfect model at double length, so harmonics are resolved in favour of the shortest period
    that explains the structure.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel() & 1
    n = int(min(bits.size, max_bits))
    if n < min_period * min_frames:
        return []
    bits = bits[:n]
    out = []
    from scipy import stats as st
    for p in range(int(min_period), min(int(max_period), n // min_frames) + 1):
        n_frames = n // p
        if n_frames < min_frames:
            break
        mat = bits[: n_frames * p].reshape(n_frames, p)
        ones = mat.sum(axis=0)
        agree = np.maximum(ones, n_frames - ones) / n_frames
        # Bonferroni-corrected binomial test per column
        try:
            pvals = st.binom.sf(ones - 1, n_frames, 0.5)
            pvals = np.minimum(pvals * p, 1.0)
            fixed_mask = (pvals < alpha) | (agree >= 0.9)
        except Exception:                                               # pragma: no cover
            sigma = 0.5 / np.sqrt(n_frames)
            fixed_mask = np.abs(agree - 0.5) > 4.0 * sigma
        n_fixed = int(np.count_nonzero(fixed_mask))
        if n_fixed < min_fixed:
            continue
        consensus = (ones >= (n_frames / 2.0)).astype(np.uint8)
        # validate the frame model against the data
        fixed_idx = np.flatnonzero(fixed_mask)
        match = (mat[:, fixed_idx] == consensus[fixed_idx])
        per_row = match.mean(axis=1)
        model_match = float(np.mean(per_row))
        frames_consistent = float(np.mean(per_row >= 0.95))
        out.append({"period_bits": int(p), "frames": int(n_frames),
                    "agreement": float(1.0 - np.mean(np.abs(agree - 1.0))),
                    "fixed_columns": n_fixed, "fixed_fraction": float(n_fixed / p),
                    "model_match": model_match, "frames_consistent": frames_consistent,
                    "consensus": consensus, "fixed_mask": fixed_mask,
                    "header_prefix": _header_prefix(agree, consensus, fixed_mask),
                    "per_row_match": per_row, "matrix": mat, "n_frames": int(n_frames)})
    if not out:
        return []
    # keep candidates whose frame model actually explains the observation
    good = [r for r in out if r["frames_consistent"] >= 0.8 and r["model_match"] >= 0.85]
    if not good:
        good = out
    good.sort(key=lambda r: r["period_bits"])
    kept = []
    for r in good:
        if any(r["period_bits"] % k["period_bits"] == 0 for k in kept):
            continue                                   # harmonic of an already accepted frame
        kept.append(r)
    kept.sort(key=lambda r: -r["fixed_columns"])
    kept = kept[:max_candidates]
    for r in kept:
        r["score"] = float(r["fixed_columns"] * r["model_match"])
        r["confidence"] = float(clamp01((r["fixed_columns"] - min_fixed) / 20.0) *
                                clamp01(r["frames"] / 8.0) *
                                clamp01(r["model_match"]))
    return kept


def _header_prefix(agree: np.ndarray, consensus: np.ndarray, fixed_mask: np.ndarray) -> dict:
    """Longest strongly-fixed prefix of the frame (the header candidate) and its pattern."""
    strong = (agree >= 0.9) & fixed_mask
    length = 0
    for c in range(strong.size):
        if strong[c]:
            length = c + 1
        else:
            break
    return {"bits": int(length), "pattern": consensus[:length].copy() if length else
            np.zeros(0, dtype=np.uint8),
            "ambiguous_after": int(np.count_nonzero(~strong[: min(strong.size, length + 8)]))}


def _consensus(mat: np.ndarray) -> np.ndarray:
    return (mat.mean(axis=0) >= 0.5).astype(np.uint8)


def auto_discover(bits: np.ndarray, min_period: int = 8, max_period: int = 512) -> dict:
    """Find periodic structure and, if the head of the frame is stable, a header candidate."""
    cands = repeated_structure(bits, min_period=min_period, max_period=max_period)
    if not cands:
        return {"ok": True, "candidates": [],
                "message": "no periodic bit structure was found in this record "
                           f"(searched frame lengths {min_period}..{max_period} bits)"}
    out = []
    for c in cands:
        hp = c["header_prefix"]
        rec = {k: v for k, v in c.items()
               if k not in ("matrix", "fixed_mask", "consensus", "header_prefix", "per_row_match")}
        rec["header_bits"] = int(hp["bits"])
        rec["header_hex"] = bits_to_hex(hp["pattern"]) if hp["bits"] else ""
        rec["ambiguous_prefix_bits"] = int(hp["ambiguous_after"])
        rec["pattern_hex"] = bits_to_hex(c["consensus"][: max(64, hp["bits"])])
        out.append(rec)
    return {"ok": True, "candidates": out, "best": out[0],
            "method": "frame-length search: the bitstream is folded at every candidate frame "
                      "length, columns are tested for significance against the binomial null with a "
                      "Bonferroni correction, and each surviving frame model is validated against "
                      "the observed frames (model_match / frames_consistent)"}


def analyse_correlation(bits: np.ndarray, patterns: list[dict] | None = None,
                        max_errors: int = 0, auto: bool = True) -> dict:
    """Run the pattern searches plus (optionally) automatic header discovery."""
    bits = np.asarray(bits, dtype=np.uint8).ravel() & 1
    if bits.size < 16:
        return {"ok": False, "searches": [],
                "message": f"only {bits.size} bits available for a correlation search"}
    searches = []
    for p in (patterns or []):
        needle = p.get("bits")
        if needle is None:
            continue
        res = find_pattern(bits, needle, max_errors=int(p.get("max_errors", max_errors)))
        res.update({"pattern": p.get("label"), "kind": p.get("kind"), "pattern_hex": p.get("hex"),
                    "max_errors": int(p.get("max_errors", max_errors))})
        searches.append(res)
    out = {"ok": True, "searches": searches, "n_bits": int(bits.size),
           "summary": _summarise(searches)}
    if auto:
        out["auto"] = auto_discover(bits)
    return out


def _summarise(searches: list[dict]) -> str:
    if not searches:
        return "no user patterns were supplied"
    parts = []
    for s in searches:
        parts.append(f"'{s.get('pattern')}': {s['n_hits']} match(es)"
                     + (" (exact)" if s.get("max_errors", 0) == 0 else
                        f" with up to {s['max_errors']} bit errors"))
    return "; ".join(parts)


def correlation_params(result: dict) -> list[dict]:
    """Parameter-record style summaries for the report."""
    out = []
    for s in (result or {}).get("searches", []):
        out.append(param(f"Pattern search '{s.get('pattern')}'", s["n_hits"], "matches",
                         confidence=1.0 if s["n_hits"] else None,
                         method=s.get("method", ""),
                         evidence=[f"{s['n_hits']} match(es) in {s['n_bits_searched']:,} bits",
                                   f"pattern length {s['needle_bits']} bits"
                                   + (f", up to {s['max_errors']} bit errors allowed"
                                      if s.get("max_errors") else ", exact match only")],
                         status="ok" if s["n_hits"] else "none"))
    auto = (result or {}).get("auto") or {}
    best = auto.get("best")
    if best:
        out.append(param("Periodic structure", f"{best['period_bits']} bit frame",
                         None, best["confidence"],
                         "frame-length search with per-column agreement scoring",
                         evidence=[f"agreement {best['agreement'] * 100:.1f} % over "
                                   f"{best['frames']} frames",
                                   f"{best['fixed_columns']} of {best['period_bits']} columns are "
                                   f"structurally fixed",
                                   f"header candidate: {best['header_bits']} bits "
                                   f"({best.get('header_hex') or 'n/a'})"],
                         limitations=["a repeating pattern is evidence of framing, not proof of a "
                                      "protocol header or of a decoded message"]))
    return out
