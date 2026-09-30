"""IQ / WAV file identification and loading.

Two responsibilities:

1. `detect_format(path)` - evidence-based identification of the container and the
   numeric sample format of a recording.  Every candidate format is *scored*
   against measurable properties of the byte stream (value distribution, float
   validity, quantisation/striding artefacts, I/Q balance, ...) and the decision
   is reported with a confidence and the evidence list that produced it.

2. `load_iq(path, spec)` - memory-bounded loader returning complex baseband (or
   real samples, flagged as such) plus structural metadata.

Nothing here ever invents metadata: absent fields are reported as
"Unknown / requires estimation".
"""
from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass, field, asdict
from typing import Any, BinaryIO

import numpy as np

from .utils import clamp01, param, STATUS_OK, STATUS_LOW, STATUS_UNAVAILABLE

# Candidate raw sample formats: (label, numpy dtype, bytes, signed/float family)
RAW_CANDIDATES: list[tuple[str, str]] = [
    ("int8", "i1"), ("uint8", "u1"),
    ("int16", "i2"), ("uint16", "u2"),
    ("int32", "i4"), ("uint32", "u4"),
    ("float32", "f4"), ("float64", "f8"),
]

WAV_FORMAT_TAGS = {0x0001: "PCM (integer)", 0x0003: "IEEE float", 0x0006: "A-law",
                   0x0007: "mu-law", 0xFFFE: "WAVE_FORMAT_EXTENSIBLE"}

# Formats unambiguous from the file extension (still validated against size)
EXTENSION_HINTS = {
    ".cf32": ("float32", "complex interleaved (GNU Radio .cf32)"),
    ".fc32": ("float32", "complex interleaved (GNU Radio .fc32)"),
    ".cs16": ("int16", "complex interleaved (GNU Radio/Magma .cs16)"),
    ".sc16": ("uint16", "complex interleaved signed-16 stored unsigned (Ettus .sc16)"),
    ".cs8": ("int8", "complex interleaved (GNU Radio .cs8)"),
    ".cu8": ("uint8", "complex interleaved (GNU Radio .cu8)"),
    ".cf64": ("float64", "complex interleaved (.cf64)"),
    ".iq": (None, "raw IQ container - sample format must be detected"),
    ".bin": (None, "raw binary container - sample format must be detected"),
    ".dat": (None, "raw binary container - sample format must be detected"),
    ".sigmf-data": (None, "SigMF recording - format in companion .sigmf-meta"),
}

_READ_LIMIT_BYTES = 96 * 1024 * 1024   # max bytes pulled for format statistics


# --------------------------------------------------------------------------- #
#  Container detection
# --------------------------------------------------------------------------- #
def sniff_container(path: str, head: bytes) -> dict:
    """Identify the container from magic bytes / extension."""
    ext = os.path.splitext(path)[1].lower()
    info: dict[str, Any] = {"extension": ext, "magic": head[:16].hex()}
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        info.update(container="wav", description="RIFF/WAVE audio container", confidence=1.0,
                    evidence=[f"RIFF....WAVE signature present ({info['magic'][:16]})"])
        return info
    if head[:6] == b"\x93NUMPY":
        info.update(container="npy", description="NumPy .npy array", confidence=1.0,
                    evidence=["NumPy magic \\x93NUMPY present"])
        return info
    if head[:6] == b"RIFX":
        info.update(container="wav", description="RIFF/WAVE (big-endian RIFX)", confidence=0.9,
                    evidence=["RIFX signature present"])
        return info
    if ext in EXTENSION_HINTS:
        hint, desc = EXTENSION_HINTS[ext]
        info.update(container="raw", description=desc, confidence=0.75 if hint else 0.4,
                    extension_hint=hint,
                    evidence=[f"File extension '{ext}' is a known raw-IQ convention"])
        return info
    if ext in (".txt", ".csv", ".tsv"):
        info.update(container="text", description="text table of numeric samples", confidence=0.8,
                    evidence=[f"extension '{ext}' indicates numeric text data"])
        return info
    # Fall back to content statistics
    printable = sum(1 for b in head[:512] if 32 <= b < 127 or b in (9, 10, 13, 32))
    if len(head) and printable / len(head) > 0.95:
        info.update(container="text", description="ASCII/UTF-8 text (numeric table assumed)",
                    confidence=0.6, evidence=["first 512 bytes are >95% printable ASCII"])
        return info
    info.update(container="raw", description="unrecognised binary container, treated as raw IQ",
                confidence=0.35,
                evidence=["no RIFF/NPY magic found", "extension is not a known IQ convention"])
    return info


def parse_wav_header(path: str, max_chunks: int = 40) -> dict:
    """Full RIFF chunk walk of a WAV file (fmt / data / LIST-INFO / bext)."""
    out: dict[str, Any] = {"valid": False, "warnings": [], "chunks": []}
    with open(path, "rb") as f:
        riff = f.read(12)
        if len(riff) < 12 or riff[:4] not in (b"RIFF", b"RIFX"):
            out["warnings"].append("missing RIFF header")
            return out
        endian = "<" if riff[:4] == b"RIFF" else ">"
        out["endian"] = endian
        pos = 12
        data_offset = data_size = None
        for _ in range(max_chunks):
            f.seek(pos)
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            cid, csize = struct.unpack(endian + "4sI", hdr)
            try:
                cname = cid.decode("ascii", "replace").strip()
            except Exception:
                cname = repr(cid)
            out["chunks"].append({"id": cname, "size": int(csize)})
            if cid == b"fmt ":
                fmt = f.read(min(csize, 64))
                (tag, channels, rate, byte_rate, block_align, bits) = struct.unpack(endian + "HHIIHH", fmt[:16])
                out.update(format_tag=tag, format_name=WAV_FORMAT_TAGS.get(tag, f"unknown (0x{tag:04x})"),
                           channels=channels, sample_rate=float(rate), byte_rate=byte_rate,
                           block_align=block_align, bits_per_sample=bits)
                if tag == 0xFFFE and len(fmt) >= 26:
                    sub = struct.unpack(endian + "H", fmt[24:26])[0]
                    out["sub_format"] = WAV_FORMAT_TAGS.get(sub, f"0x{sub:04x}")
                if csize % 2:
                    csize += 1
            elif cid == b"data":
                data_offset, data_size = pos + 8, int(csize)
            elif cid in (b"LIST", b"INFO"):
                blob = f.read(min(csize, 512))
                info_pairs: dict[str, str] = {}
                i = 0
                while i + 8 <= len(blob):
                    sid = blob[i:i + 4]
                    ssize = struct.unpack(endian + "I", blob[i + 4:i + 8])[0]
                    val = blob[i + 8:i + 8 + min(ssize, 128)]
                    info_pairs[sid.decode("ascii", "replace")] = val.decode("ascii", "replace").strip("\x00 ")
                    i += 8 + ssize + (ssize % 2)
                out["info"] = info_pairs
            elif cid == b"bext":
                blob = f.read(min(csize, 256))
                out["bext_preview"] = blob[:48].decode("ascii", "replace").strip("\x00 ")
            pos += 8 + csize + (csize % 2)
        out["data_offset"] = data_offset
        out["data_size"] = data_size
        out["valid"] = data_offset is not None and out.get("channels") is not None
        if data_offset is not None and out.get("block_align"):
            n_frames = out["data_size"] // out["block_align"]
            out["n_frames"] = int(n_frames)
            out["duration_s"] = float(n_frames / out["sample_rate"]) if out.get("sample_rate") else None
        if not out["valid"]:
            out["warnings"].append("WAV has no usable fmt/data chunk")
    return out


# --------------------------------------------------------------------------- #
#  Raw format scoring
# --------------------------------------------------------------------------- #
@dataclass
class FormatCandidate:
    dtype: str
    endian: str
    score: float = 0.0
    evidence: list[str] = field(default_factory=list)
    penalties: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        name = {"i1": "int8", "u1": "uint8", "i2": "int16", "u2": "uint16",
                "i4": "int32", "u4": "uint32", "f4": "float32", "f8": "float64"}[self.dtype]
        return name

    def to_dict(self) -> dict:
        d = asdict(self)
        d["name"] = self.label
        d["endian_name"] = "little" if self.endian == "<" else "big"
        return d


def _dtype_str(code: str, endian: str) -> str:
    if code.startswith(("i", "u", "f")) and code[1].isdigit() and endian == ">":
        return ">" + code
    return code


def _score_candidate(buf: np.ndarray, code: str, endian: str, file_bytes: int,
                     extension_hint: str | None) -> FormatCandidate:
    """Score how plausible it is that `buf` (raw uint8) holds samples of dtype code."""
    cand = FormatCandidate(code, endian)
    dt = np.dtype(_dtype_str(code, endian))
    item = dt.itemsize
    if file_bytes < item * 8:
        cand.penalties.append("file smaller than 8 samples of this format")
        cand.score = 0.0
        return cand
    n_vals = buf.size // item
    if n_vals < 16:
        cand.penalties.append("too few 8-bit groups to analyse")
        cand.score = 0.0
        return cand
    vals = buf[: n_vals * item].view(dt).astype(np.float64)
    if endian == ">" and item > 1:
        vals = vals.copy()

    n_used = min(vals.size, 400_000)
    v = vals[:n_used]
    is_float = dt.kind == "f"
    metrics: dict[str, float] = {"n_values": int(vals.size)}
    score = 0.0
    ev, pen = cand.evidence, cand.penalties

    if is_float:
        finite = np.isfinite(v)
        finite_frac = float(finite.mean())
        metrics["finite_fraction"] = round(finite_frac, 6)
        if finite_frac < 0.99:
            pen.append(f"{100*(1-finite_frac):.2f}% of values are NaN/Inf - implausible sample values")
            score -= 0.60 * (1 - finite_frac) * 10
        else:
            ev.append(f"all {n_used} inspected values are finite floats")
            score += 0.35
        vf = v[finite]
        if vf.size:
            absv = np.abs(vf)
            metrics["p_abs_lt_1"] = round(float((absv <= 1.5).mean()), 4)
            metrics["p99_abs"] = round(float(np.percentile(absv, 99)), 6)
            if metrics["p_abs_lt_1"] > 0.99:
                ev.append("99% of magnitudes within +/-1.5 (normalised float convention)")
                score += 0.25
            elif metrics["p_abs_lt_1"] > 0.95:
                ev.append("most magnitudes within +/-1.5 (normalised float convention)")
                score += 0.12
            if np.percentile(absv, 99) < 1e-4 and np.percentile(absv, 99) > 0:
                pen.append("float values concentrated near zero (likely mis-typed integer bytes)")
                score -= 0.3
            uniq = len(np.unique(np.round(absv[:20000], 12)))
            metrics["approx_unique_frac"] = round(uniq / max(1, min(20000, absv.size)), 4)
            if uniq > 0.9 * min(20000, absv.size):
                ev.append("value spread is continuous (no quantisation grid) - typical of float captures")
                score += 0.15
    else:
        info = np.iinfo(dt)
        vmin, vmax = float(np.min(v)), float(np.max(v))
        p1, p99 = float(np.percentile(v, 1)), float(np.percentile(v, 99))
        span = info.max - info.min + 1
        metrics.update(min=vmin, max=vmax, p1=p1, p99=p99)
        util = (p99 - p1) / span
        metrics["range_utilisation"] = round(util, 5)
        # degenerate distributions (a clear sign of a wrong word size)
        zero_frac = float((v == 0).mean())
        const_frac = float((v == vmin).mean())
        metrics["exact_zero_fraction"] = round(zero_frac, 4)
        metrics["extreme_value_fraction"] = round(const_frac, 4)
        if zero_frac > 0.30:
            pen.append(f"{100*zero_frac:.1f}% of values are exactly 0 (byte-stride artefact of a wider integer word)")
            score -= 0.45
        if util < 0.004:
            pen.append("uses <0.4% of the representable range - wrong word size")
            score -= 0.35
        elif util < 0.05:
            pen.append("uses <5% of the representable range")
            score -= 0.10
        elif util <= 1.0:
            ev.append(f"values occupy {100*util:.1f}% of the format range (plausible signal occupancy)")
            score += 0.22
        # a random byte reinterpretation of a wider format tends to look uniform
        hist, _ = np.histogram(v, bins=64, range=(info.min, info.max + 1))
        hist = hist / max(hist.sum(), 1)
        nz = hist[hist > 0]
        flatness = float(np.std(nz) / max(np.mean(nz), 1e-12)) if nz.size else 9.9
        metrics["histogram_flatness"] = round(flatness, 4)
        if flatness < 0.35 and util > 0.5:
            pen.append("near-uniform value histogram - characteristic of misinterpreted bytes, not of a real signal")
            score -= 0.40
        elif flatness < 0.9:
            ev.append("value histogram is non-uniform (central clustering), consistent with a real baseband capture")
            score += 0.18
        # quantisation-stride test: is every value a multiple of 2^8, 2^16 ... ?
        for shift in (8, 16):
            if info.max >= (1 << (8 + shift)):
                bits_low = (v.astype(np.int64) & ((1 << shift) - 1))
                frac_zero = float((bits_low == 0).mean())
                if frac_zero > 0.98:
                    pen.append(f"low {shift} bits are zero in >98% of samples - data is really a wider word")
                    score -= 0.5
                    break
    # spectral / distribution plausibility of *this* interpretation
    if n_vals >= 4096 or vals.size >= 4096:
        sub = vals[: min(vals.size, 400_000)]
        if sub.size >= 4096:
            s_sp, ev_sp, m_sp = _spectral_probe(sub)
            s_di, ev_di, m_di = _distribution_shape_probe(sub)
            score += s_sp + s_di
            ev += ev_sp + ev_di
            metrics.update(m_sp)
            metrics.update(m_di)
        else:
            s_di, ev_di, m_di = _distribution_shape_probe(sub)
            score += s_di
            ev += ev_di
            metrics.update(m_di)
    # striding / word-alignment evidence
    if file_bytes % item == 0:
        ev.append(f"file size {file_bytes} is an exact multiple of {item} bytes")
        score += 0.05
    else:
        pen.append(f"file size not divisible by {item} (trailing partial sample)")
        score -= 0.05
    if extension_hint and cand.label == extension_hint:
        ev.append(f"sample format agrees with the file-extension convention for this layout")
        score += 0.30
    cand.score = score
    cand.metrics = metrics
    return cand


def _spectral_probe(vals: np.ndarray, fs_norm: float = 1.0) -> tuple[float, list[str], dict]:
    """Measure spectral concentration of a candidate interpretation.

    A correctly interpreted complex baseband recording concentrates most of its
    power in a fraction of the Nyquist band and shows a band edge; a byte-level
    misinterpretation produces white, flat spectrum filling the whole band.  This
    is the single most selective discriminator between competing word sizes.
    """
    from scipy import signal as sigproc
    n = (vals.size // 2) * 2
    if n < 4096:
        return 0.0, [], {}
    x = vals[:n:2] + 1j * vals[1:n:2]
    nperseg = int(min(4096, max(1024, x.size // 8)))
    try:
        f, P = sigproc.welch(x, fs=fs_norm, nperseg=nperseg, return_onesided=False)
    except Exception:
        return 0.0, [], {}
    P = np.asarray(P, dtype=np.float64)
    if not np.all(np.isfinite(P)) or P.sum() <= 0:
        return -0.5, ["spectrum of this interpretation is degenerate"], {}
    order = np.argsort(f)
    f, P = f[order], P[order]
    PdB = 10 * np.log10(np.maximum(P, P.max() * 1e-12))
    med = float(np.median(PdB))
    peak_to_med = float(np.max(PdB) - med)
    total = P.sum()
    c = np.cumsum(P) / total
    i_lo, i_hi = int(np.searchsorted(c, 0.05)), int(np.searchsorted(c, 0.95))
    occ = float((f[i_hi] - f[i_lo]) / (f.max() - f.min() + 1e-12))
    edge_sharp = float(np.percentile(PdB, 95) - med)
    m = {"peak_to_median_db": round(peak_to_med, 3), "90pct_occupancy": round(occ, 4),
         "spectral_edge_db": round(edge_sharp, 3)}
    ev: list[str] = []
    score = 0.0
    if occ < 0.45 and edge_sharp > 3.0:
        ev.append(f"power concentrated in {100*occ:.0f}% of the Nyquist band with a defined spectral edge "
                  f"({edge_sharp:.1f} dB) - consistent with a modulated carrier")
        score += 0.45
    elif occ < 0.7:
        ev.append(f"power occupies {100*occ:.0f}% of the Nyquist band")
        score += 0.15
    else:
        ev.append(f"power fills {100*occ:.0f}% of the Nyquist band (white-like)")
        score -= 0.45
    if peak_to_med > 12:
        ev.append(f"peak-to-median spectral contrast {peak_to_med:.1f} dB")
        score += 0.10
    return score, ev, m


def _distribution_shape_probe(vals: np.ndarray) -> tuple[float, list[str], dict]:
    """Peakedness of the sample distribution: real signals cluster, mis-typed bytes do not."""
    if vals.size < 512:
        return 0.0, [], {}
    v = vals[: min(vals.size, 200_000)]
    sd = float(v.std())
    if sd <= 0:
        return -0.4, ["sample values are constant"], {"excess_kurtosis": None}
    z = (v - v.mean()) / sd
    kurt = float(np.mean(z ** 4) - 3.0)
    hist, _ = np.histogram(z, bins=32, range=(-4, 4))
    frac = hist / max(hist.sum(), 1)
    nz = frac[frac > 0]
    flatness = float(np.std(nz) / max(np.mean(nz), 1e-12)) if nz.size else 9.9
    m = {"excess_kurtosis": round(kurt, 3), "histogram_flatness": round(flatness, 3)}
    ev, score = [], 0.0
    if kurt < -1.0:
        ev.append(f"sample distribution is near-uniform (excess kurtosis {kurt:.2f}) - implausible for a "
                  f"real baseband digitisation")
        score -= 0.45
    elif kurt > -0.5:
        ev.append(f"sample distribution is centrally peaked (excess kurtosis {kurt:.2f}) - typical of real "
                  f"baseband IQ")
        score += 0.22
    return score, ev, m


def _iq_balance_score(vals: np.ndarray) -> tuple[float, list[str], dict]:
    """I/Q structural consistency: power balance, distribution similarity, correlation."""
    ev: list[str] = []
    if vals.size < 64:
        return 0.0, ["too few samples for I/Q checks"], {}
    # Arbitrary bytes can decode to NaN/Inf under a float hypothesis (random data), which made the
    # histogram below raise "range is not finite" - a hard error on input that must be *scored*.
    # Non-finite values are excluded and their fraction is reported as evidence instead.
    finite = np.isfinite(vals)
    n_finite = int(np.count_nonzero(finite))
    if n_finite < 64:
        return -0.3, [f"only {n_finite} of {vals.size} values are finite under this hypothesis "
                      f"(NaN/Inf present)"], {"finite_fraction": round(n_finite / max(vals.size, 1), 4)}
    if n_finite < vals.size:
        ev.append(f"{vals.size - n_finite} of {vals.size} values are non-finite (NaN/Inf): a strong "
                  f"indication against this float format")
    vals = vals[finite]
    n = (vals.size // 2) * 2
    i, q = vals[:n:2], vals[1:n:2]
    if i.size < 16:
        return 0.0, ["too few finite samples for I/Q checks"], {"finite_fraction": 1.0}
    pi, pq = float(np.mean(np.abs(i))), float(np.mean(np.abs(q)))
    if pi <= 0 or pq <= 0:
        return -0.4, ["one of the I/Q streams is identically zero"], {"power_ratio": None}
    ratio_db = 10 * math.log10(pi / pq)
    score = 0.0
    m: dict[str, Any] = {"iq_power_ratio_db": round(ratio_db, 3)}
    if abs(ratio_db) < 1.0:
        ev.append(f"I/Q power balance within {abs(ratio_db):.2f} dB")
        score += 0.22
    elif abs(ratio_db) < 3.0:
        ev.append(f"I/Q power balance within {abs(ratio_db):.2f} dB (usable, front-end imbalance possible)")
        score += 0.10
    else:
        ev.append(f"I/Q power imbalance {abs(ratio_db):.2f} dB")
        score -= 0.15
    # distribution similarity via normalised histograms
    def _hist(x: np.ndarray) -> np.ndarray:
        lo, hi_ = float(np.min(x)), float(np.max(x))
        if not (math.isfinite(lo) and math.isfinite(hi_)) or hi_ <= lo:
            return np.ones(48, dtype=np.float64) / 48.0
        h, _ = np.histogram(x, bins=48, range=(lo, hi_), density=False)
        return h / max(float(h.sum()), 1.0)

    hi = _hist(i)
    hq = _hist(q)
    l1 = float(np.sum(np.abs(hi - hq)) / 2.0)
    m["iq_histogram_l1_distance"] = round(l1, 4)
    if l1 < 0.25:
        ev.append(f"I and Q amplitude distributions agree (L1 distance {l1:.3f})")
        score += 0.15
    rho = float(np.corrcoef(i[: min(20000, i.size)], q[: min(20000, q.size)])[0, 1]) if i.size > 8 else 0.0
    m["iq_correlation"] = round(rho, 4)
    if abs(rho) < 0.3:
        ev.append(f"I/Q near-independent (|rho|={abs(rho):.3f}) - consistent with complex baseband")
        score += 0.12
    else:
        ev.append(f"I/Q correlated (rho={rho:.3f}) - possible real/duplicated stream or strong phase structure")
        score -= 0.20
    return score, ev, m


def detect_raw_format(path: str, container: dict) -> dict:
    """Score every plausible raw sample format and return a ranked decision."""
    file_bytes = os.path.getsize(path)
    with open(path, "rb") as f:
        raw = f.read(min(file_bytes, _READ_LIMIT_BYTES))
    buf = np.frombuffer(raw, dtype=np.uint8)
    endians = ["<", ">"] if container.get("extension") not in (".cf32", ".cs16", ".fc32") else ["<", ">"]
    cands: list[FormatCandidate] = []
    for name, code in RAW_CANDIDATES:
        for en in endians:
            c = _score_candidate(buf, code, en, file_bytes, container.get("extension_hint"))
            if c.score <= -9:
                continue
            cands.append(c)
    cands.sort(key=lambda c: -c.score)
    top = cands[:4]
    best: list[FormatCandidate] = []
    for c in top:
        dt = np.dtype(_dtype_str(c.dtype, c.endian))
        item = dt.itemsize
        if file_bytes // item < 64:
            continue
        vals = buf[: (buf.size // item) * item].view(dt)
        if item > 500_000:  # guard against pathological dtypes
            continue
        sub = vals[: min(vals.size, 400_000)].astype(np.float64)
        s, ev, metrics = _iq_balance_score(sub)
        c.score += s
        c.evidence += ev
        c.metrics.update(metrics)
        best.append(c)
    if not best:
        best = top
    best.sort(key=lambda c: -c.score)
    top = best
    # Confidence: normalised softmax over the plausible candidates + absolute quality
    scores = np.array([max(c.score, -1.0) for c in top], dtype=np.float64)
    if scores.size == 1:
        conf = 0.75
    else:
        w = np.exp((scores - scores.max()) / 0.35)
        conf = float(w[0] / w.sum())
    conf *= clamp01(0.55 + 0.35 * max(0.0, min(1.0, top[0].score / 0.9)))
    return {
        "format": top[0].label,
        "dtype": _dtype_str(top[0].dtype, top[0].endian),
        "endian": top[0].endian,
        "endian_name": "little" if top[0].endian == "<" else "big",
        "confidence": round(float(conf), 4),
        "evidence": top[0].evidence,
        "limitations": top[0].penalties,
        "metrics": top[0].metrics,
        "candidates": [
            {"format": c.label, "dtype": _dtype_str(c.dtype, c.endian),
             "endian": c.endian, "endian_name": "little" if c.endian == "<" else "big",
             "score": round(float(c.score), 4),
             "confidence": round(float(np.exp((c.score - top[0].score) / 0.35) /
                                       np.sum(np.exp((np.array([x.score for x in top]) - top[0].score) / 0.35))), 4),
             "evidence": c.evidence, "rejections": c.penalties, "metrics": c.metrics}
            for c in top
        ],
        "bytes_inspected": int(buf.size),
    }


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def detect_format(path: str, filename: str | None = None) -> dict:
    """Full identification of a recording: container + sample format + structure.

    Detection runs on arbitrary bytes, including files that are not signals at all (a text file
    renamed to .wav, a truncated dump).  Every numeric warning is therefore suppressed inside this
    function: a corrupt input must produce a *low-confidence detection with a reason*, never a
    traceback or a wall of numpy warnings.
    """
    import warnings as _warnings
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                return _detect_format_impl(path, filename)
    except Exception as exc:
        # Identification runs on arbitrary bytes: a truncated header, a zero-length file or a
        # container whose declared sizes do not match reality must produce a *detection result that
        # says so*, never an exception.  (Measured: a 52-byte pseudo-WAV header made the header
        # parser raise struct.error, which surfaced as a 500 on upload.)
        name = filename or os.path.basename(path)
        try:
            size = os.path.getsize(path)
        except OSError:                                                  # pragma: no cover
            size = 0
        return {
            "filename": name, "file_size_bytes": int(size), "container": "invalid",
            "container_description": (f"the file could not be parsed as a signal container: "
                                      f"{type(exc).__name__}: {exc}"),
            "container_confidence": 0.0, "confidence": 0.0, "extension": os.path.splitext(name)[1].lower(),
            "dtype": None, "format": None, "is_complex": False, "iq_layout": None, "channels": None,
            "bits_per_sample": None, "n_samples": 0, "n_frames": 0, "duration_s": None,
            "sample_rate": None, "sample_rate_source": "Unknown / requires estimation",
            "center_frequency": None,
            "center_frequency_source": "Unknown / requires estimation",
            "detection_error": f"{type(exc).__name__}: {exc}",
            "evidence": [f"the container parser failed on this byte stream: "
                         f"{type(exc).__name__}: {exc}",
                         f"file size {size} bytes"],
            "limitations": ["no sample format, structure or metadata could be determined from this "
                            "file: it is not a supported IQ/WAV recording"],
            "manual_override_hint": ("if the file really is a raw capture, name the dtype, byte order "
                                     "and I/Q layout explicitly and re-upload"),
        }


def _detect_format_impl(path: str, filename: str | None = None) -> dict:
    filename = filename or os.path.basename(path)
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(4096)
    container = sniff_container(filename, head)
    result: dict[str, Any] = {
        "filename": filename,
        "file_size_bytes": int(size),
        "container": container["container"],
        "container_description": container["description"],
        "container_confidence": container["confidence"],
        "container_evidence": container.get("evidence", []),
        "extension": container.get("extension"),
        "is_complex": True,
        "iq_layout": "interleaved I/Q (I even, Q odd)",
        "sample_rate": None,
        "sample_rate_source": "Unknown / requires estimation",
        "center_frequency": None,
        "center_frequency_source": "Unknown / requires estimation",
        "channels": None,
        "warnings": [],
    }

    if container["container"] == "wav":
        wav = parse_wav_header(path)
        result["wav"] = wav
        if not wav.get("valid"):
            result["warnings"] += wav.get("warnings", []) + ["WAV container could not be parsed"]
            result["format"] = "invalid WAV"
            result["confidence"] = 0.0
            result["evidence"] = wav.get("warnings", [])
            result["candidates"] = []
            result["dtype"] = None
            result["error"] = "invalid_wav"
            return result
        ch = int(wav["channels"]); bits = int(wav["bits_per_sample"])
        fmt_is_float = wav.get("format_tag") == 3 or wav.get("sub_format", "").startswith("IEEE")
        if bits == 64:
            base = "float64"
        elif bits == 32:
            base = "float32" if fmt_is_float else "int32"
        elif bits == 16:
            base = "int16"
        elif bits == 8:
            base = "uint8"
        elif bits == 24:
            base = "int24"
        else:
            base = f"int{bits}"
        result.update(
            sample_rate=float(wav["sample_rate"]),
            sample_rate_source="WAV fmt chunk",
            channels=ch,
            bits_per_sample=bits,
            n_frames=wav.get("n_frames"),
            duration_s=wav.get("duration_s"),
            dtype=base,
            endian=wav.get("endian", "<"),
            format=f"{ch}ch {base} WAV",
            evidence=[
                f"RIFF/WAVE header, format tag {wav.get('format_name')}",
                f"fmt chunk declares {ch} channel(s) @ {wav['sample_rate']:.0f} Hz, {bits} bit",
                f"data chunk holds {wav.get('n_frames')} frames ({wav.get('duration_s', 0):.3f} s)",
            ],
        )
        if wav.get("info"):
            result["metadata_info"] = wav["info"]
        if ch >= 2:
            # complex IQ vs stereo audio: decide from rate, format tag and content
            iq_votes, audio_votes, why = 0, 0, []
            if wav["sample_rate"] >= 200_000:
                iq_votes += 2; why.append(f"sample rate {wav['sample_rate']:.0f} Hz is far above audio (>=200 kHz)")
            elif wav["sample_rate"] >= 96_000:
                iq_votes += 1; why.append(f"sample rate {wav['sample_rate']:.0f} Hz is above standard audio")
            else:
                audio_votes += 2; why.append(f"sample rate {wav['sample_rate']:.0f} Hz is a standard audio rate")
            if fmt_is_float:
                iq_votes += 1; why.append("IEEE-float samples are typical of SDR/analytic captures")
            mean_v, corr = _wav_iq_probe(path, wav, ch)
            result["iq_probe"] = {"mean_correlation_LR": corr, "samples": mean_v}
            if corr is not None:
                if abs(corr) < 0.2:
                    iq_votes += 1; why.append(f"L/R channels near-uncorrelated (rho={corr:.3f}) - typical of I/Q")
                else:
                    audio_votes += 1; why.append(f"L/R channels correlated (rho={corr:.3f}) - typical of stereo audio")
            result["is_complex"] = iq_votes > audio_votes
            total = max(iq_votes + audio_votes, 1)
            result["confidence"] = round(0.6 + 0.4 * abs(iq_votes - audio_votes) / total, 3)
            result["evidence"] += why
            result["iq_layout"] = ("interleaved I/Q (L=I, R=Q)" if result["is_complex"]
                                   else "stereo real audio (not analytic IQ)")
            if not result["is_complex"]:
                result["warnings"].append(
                    "This WAV looks like stereo audio rather than complex IQ; the platform will treat it as a "
                    "real passband recording unless you override the interpretation.")
        else:
            result["is_complex"] = False
            result["confidence"] = 0.95
            result["iq_layout"] = "single channel real-valued (no quadrature)"
            result["evidence"].append("mono WAV: real-valued samples, analytic IQ is not present")
            result["warnings"].append("mono recording - complex-domain metrics (IQ imbalance, constellation "
                                      "rotation) are not applicable")
        result["candidates"] = [{
            "format": result["format"], "confidence": result["confidence"],
            "evidence": result["evidence"], "rejections": [],
        }]
        return result

    if container["container"] == "npy":
        try:
            arr = np.load(path, mmap_mode="r")
            result.update(container="npy", dtype=str(arr.dtype), format=f"NumPy array {arr.dtype} shape {arr.shape}",
                          confidence=0.97,
                          is_complex=np.iscomplexobj(arr),
                          n_samples=int(arr.size if arr.ndim == 1 else arr.shape[0]),
                          evidence=["NumPy header parsed",
                                    f"array dtype {arr.dtype}, shape {arr.shape}"])
            if arr.dtype.itemsize == 1:
                result["container"] = "npy"
            if arr.ndim == 2 and arr.shape[1] == 2:
                result["iq_layout"] = "2-column array (column 0 = I, column 1 = Q)"
            if np.iscomplexobj(arr):
                result["iq_layout"] = "native complex dtype (I and Q packed by NumPy)"
            result["sample_rate"] = None
            result["sample_rate_source"] = "Unknown / requires estimation"
            return result
        except Exception as exc:                       # pragma: no cover
            result["warnings"].append(f"NPY parse failed: {exc}")

    if container["container"] == "text":
        try:
            data = np.loadtxt(path, delimiter="," if filename.lower().endswith(".csv") else None,
                              max_rows=200000, ndmin=2)
            if data.ndim == 2 and data.shape[1] >= 2:
                result.update(container="text", dtype="float64", format=f"text table, {data.shape[1]} columns",
                              confidence=0.9, is_complex=True,
                              iq_layout="column 0 = I, column 1 = Q",
                              n_samples=int(data.shape[0]),
                              evidence=[f"parsed {data.shape[0]} rows x {data.shape[1]} numeric columns",
                                        "two columns interpreted as I(sample), Q(sample)"])
            else:
                result.update(container="text", dtype="float64", format="text table, single column (real)",
                              confidence=0.7, is_complex=False, n_samples=int(data.size),
                              iq_layout="real-valued text samples",
                              evidence=[f"parsed {data.size} numeric values"])
            result["sample_rate_source"] = "Unknown / requires estimation"
            return result
        except Exception as exc:
            result["warnings"].append(f"text parse failed: {exc}")

    # ---- raw binary path ----
    det = detect_raw_format(path, container)
    itemsize = np.dtype(det["dtype"]).itemsize
    n_values = size // itemsize
    n_samples = n_values // 2 if n_values % 2 == 0 else n_values // 2
    result.update(
        format=f"Complex {det['format']} IQ" + ("" if det["endian"] == "<" else " (big-endian)"),
        dtype=det["dtype"], endian=det["endian"], confidence=det["confidence"],
        evidence=det["evidence"], limitations=det["limitations"], metrics=det["metrics"],
        candidates=det["candidates"],
        is_complex=True,
        iq_layout="interleaved I/Q (I even index, Q odd index)" if det["format"] != "float64" else
                  "interleaved I/Q (I even index, Q odd index)",
        n_samples=int(n_samples),
        sample_rate_source="Unknown / requires estimation",
    )
    if n_values % 2:
        result["warnings"].append(f"odd number of sample values ({n_values}): last value has no I/Q partner "
                                  "and will be discarded when loading")
    if det["confidence"] < 0.5:
        result["warnings"].append(
            "format confidence is low - verify with the manual override before trusting the analysis")
    if not det["evidence"]:
        result["warnings"].append("no positive evidence found for any sample format")
    return result


def _wav_iq_probe(path: str, wav: dict, channels: int) -> tuple[float | None, float | None]:
    """Read a fraction of a WAV's samples and compute L/R correlation."""
    from scipy.io import wavfile
    try:
        with open(path, "rb") as f:
            if wav.get("data_offset") is None:
                return None, None
        fs, data = wavfile.read(path)
        if data.ndim != 2:
            return None, None
        n = min(data.shape[0], 200_000)
        a = data[:n, 0].astype(np.float64)
        b = data[:n, 1].astype(np.float64)
        if a.size < 64 or a.std() == 0 or b.std() == 0:
            return float(a.size), None
        rho = float(np.corrcoef(a, b)[0, 1])
        return float(a.size), rho
    except Exception:
        return None, None


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #
def load_iq(path: str, spec: dict | None = None, max_samples: int | None = None,
            offset_samples: int = 0) -> dict:
    """Load samples according to `spec` (as returned by detect_format, possibly edited).

    Returns dict with keys: samples (complex64 or float64), fs, is_complex,
    n_samples, truncated, load_notes.
    """
    spec = spec or detect_format(path)
    notes: list[str] = []
    container = spec.get("container", "raw")

    if container == "wav":
        from scipy.io import wavfile
        fs, data = wavfile.read(path)
        fs = float(fs)
        if data.ndim == 1:
            x = data.astype(np.float64)
            is_complex = False
        else:
            if spec.get("is_complex", True):
                x = data[:, 0].astype(np.float64) + 1j * data[:, 1].astype(np.float64)
                is_complex = True
            else:
                x = data[:, 0].astype(np.float64)
                is_complex = False
                notes.append("stereo WAV treated as real passband (left channel only)")
        if data.dtype.kind == "i":
            scale = float(np.iinfo(data.dtype).max)
            x = x / scale
            notes.append(f"integer PCM scaled by 1/{scale:g} to +/-1.0")
        if offset_samples:
            x = x[offset_samples:]
        if max_samples and x.size > max_samples:
            x = x[:max_samples]
            notes.append(f"truncated to {max_samples} samples by the loader/session limit")
        return {"samples": x.astype(np.complex64) if is_complex else x.astype(np.float64),
                "fs": fs, "is_complex": is_complex, "n_samples": int(x.size),
                "truncated": bool(max_samples and x.size >= max_samples), "load_notes": notes}

    if container == "npy":
        arr = np.load(path, mmap_mode="r")
        if arr.ndim == 2 and arr.shape[1] == 2:
            x = np.asarray(arr[:, 0], dtype=np.float64) + 1j * np.asarray(arr[:, 1], dtype=np.float64)
            is_complex = True
        elif np.iscomplexobj(arr):
            x = np.asarray(arr, dtype=np.complex128)
            is_complex = True
        else:
            x = np.asarray(arr, dtype=np.float64).ravel()
            is_complex = False
        if offset_samples:
            x = x[offset_samples:]
        if max_samples and x.size > max_samples:
            x = x[:max_samples]
            notes.append(f"truncated to {max_samples} samples")
        return {"samples": x.astype(np.complex64) if is_complex else x, "fs": None,
                "is_complex": is_complex, "n_samples": int(x.size), "truncated": False, "load_notes": notes}

    if container == "text":
        data = np.loadtxt(path, delimiter="," if path.lower().endswith(".csv") else None, ndmin=2)
        if data.shape[1] >= 2:
            x = data[:, 0] + 1j * data[:, 1]
            is_complex = True
        else:
            x = data.ravel().astype(np.float64)
            is_complex = False
        if offset_samples:
            x = x[offset_samples:]
        if max_samples and x.size > max_samples:
            x = x[:max_samples]
        return {"samples": x.astype(np.complex64) if is_complex else x, "fs": None,
                "is_complex": is_complex, "n_samples": int(x.size), "truncated": False, "load_notes": notes}

    # raw
    fmt = spec.get("format") or ""
    dtype_map = {"int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
                 "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8"}
    name = None
    for k in dtype_map:
        if k in fmt.lower():
            name = k
            break
    dtype_code = spec.get("dtype") or (dtype_map.get(name or "int16", "i2"))
    endian = spec.get("endian", "<")
    if isinstance(dtype_code, str) and dtype_code and dtype_code[0] in "<>":
        endian = dtype_code[0]
        dtype_code = dtype_code[1:]
    dt = np.dtype(dtype_code)
    if dt.itemsize > 1 and endian == ">":
        dt = dt.newbyteorder(">")
    interleaved = spec.get("interleaved", True)
    itemsize = dt.itemsize
    # memory-bounded read
    start_byte = offset_samples * (2 if interleaved else 1) * itemsize
    if max_samples:
        n_bytes = int(max_samples) * (2 if interleaved else 1) * itemsize
    else:
        n_bytes = os.path.getsize(path) - start_byte
    arr = np.fromfile(path, dtype=dt, count=int(max(0, n_bytes // itemsize)), offset=int(start_byte))
    arr = arr.astype(np.float64)
    n_vals = arr.size
    if interleaved:
        if n_vals % 2:
            arr = arr[:-1]
            notes.append("dropped a trailing unpaired sample value")
        x = arr[0::2] + 1j * arr[1::2]
    else:
        half = n_vals // 2
        x = arr[:half] + 1j * arr[half:2 * half]
        notes.append("separate (non-interleaved) I/Q blocks: first half = I, second half = Q")
    is_complex = True
    if dt.kind == "f":
        notes.append(f"loaded as {dt} (float samples used as-is; no integer scaling applied)")
    else:
        info = np.iinfo(dt)
        if dt.kind == "u":
            mid = (int(info.max) + 1) / 2.0
            x = (x - mid) / mid
            notes.append(f"{name} unsigned integer samples re-centred around {mid:g} and scaled to +/-1.0")
        else:
            scale = float(int(info.max))
            x = x / scale
            notes.append(f"{name} signed integer samples scaled by 1/{scale:g} to +/-1.0")
    if max_samples and x.size >= max_samples:
        notes.append(f"loader cap reached: first {max_samples} samples analysed")
    return {"samples": x.astype(np.complex64), "fs": spec.get("sample_rate"),
            "is_complex": is_complex, "n_samples": int(x.size),
            "truncated": bool(max_samples and x.size >= max_samples), "load_notes": notes}


def file_report(spec: dict) -> list[dict]:
    """Field-by-field summary shown after upload (SIH §2)."""
    f = lambda n, v, unit=None, src=None: {
        "name": n, "value": v, "unit": unit,
        "source": src or ("measured" if v not in (None, "") else "Unknown / requires estimation"),
    }
    return [
        f("Filename", spec.get("filename")),
        f("File size", spec.get("file_size_bytes"), "bytes"),
        f("Container", spec.get("container_description") or spec.get("container")),
        f("Detected format", spec.get("format")),
        f("Sample format", spec.get("dtype")),
        f("Byte order", spec.get("endian_name")),
        f("I/Q structure", spec.get("iq_layout")),
        f("Channels", spec.get("channels")),
        f("Sample count", spec.get("n_samples"), "samples"),
        f("Duration", round(spec["duration_s"], 6) if spec.get("duration_s") else None, "s"),
        f("Sample rate", spec.get("sample_rate"), "Hz", spec.get("sample_rate_source")),
        f("Center frequency", spec.get("center_frequency"), "Hz", spec.get("center_frequency_source")),
        f("Format confidence", round(100 * spec.get("confidence", 0), 1), "%"),
    ]
