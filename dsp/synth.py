"""Synthetic RF signal generator with full ground truth.

Used for: (1) bundled demo data, (2) the self-benchmark ground truth,
(3) generating the training set for the classical-ML modulation classifier,
(4) the in-browser "Synthetic Generator" page.

Modulations: BPSK, QPSK, 8PSK, 16QAM, 64QAM (RRC shaped), 2FSK/GFSK (CPFSK with
Gaussian frequency pulse), AM (DSB tone/message) and FM (mono passband, plus
FM broadcast stereo MPX as a complex-IQ WAV).  Optional convolutional FEC and
interleaving, carrier/phase offset, IQ imbalance, DC offset, AWGN and burst
errors.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
from scipy import signal as sigproc

from . import bitstream as bsu

# --------------------------------------------------------------------------- #
#  Constellations (Gray-labelled) and mappings
# --------------------------------------------------------------------------- #
def _norm(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.complex128)
    return c / math.sqrt(float(np.mean(np.abs(c) ** 2)))


def _qam_levels(bits_per_axis: int) -> np.ndarray:
    """Gray-ordered PAM levels, e.g. 2 bits -> [-3,-1,+1,+3]."""
    levels = np.arange(2 ** bits_per_axis)
    gray = levels ^ (levels >> 1)                     # binary -> gray
    order = np.argsort(gray)                          # gray index -> binary index
    return (2.0 * order - (2 ** bits_per_axis - 1))


def constellation(mod: str) -> np.ndarray:
    m = mod.upper()
    if m == "BPSK":
        return _norm(np.array([-1, 1], dtype=np.complex128))
    if m == "QPSK":
        pts = np.exp(1j * (np.pi / 4 + np.arange(4) * np.pi / 2))
        return _norm(pts[[0, 2, 3, 1]])               # Gray order (00,01,10,11)->index map
    if m == "8PSK":
        idx = np.arange(8)
        gray = idx ^ (idx >> 1)
        pts = np.exp(1j * 2 * np.pi * gray / 8)
        return _norm(pts)
    if m == "16QAM":
        levels = _qam_levels(2)
        g = np.meshgrid(levels, levels)
        pts = (g[0] + 1j * g[1]).ravel()
        return _norm(pts)
    if m == "64QAM":
        levels = _qam_levels(3)
        g = np.meshgrid(levels, levels)
        pts = (g[0] + 1j * g[1]).ravel()
        return _norm(pts)
    raise ValueError(f"unsupported modulation for constellation: {mod}")


def bits_per_symbol(mod: str) -> int:
    m = mod.upper()
    return {"BPSK": 1, "QPSK": 2, "8PSK": 3, "16QAM": 4, "64QAM": 6, "2FSK": 1, "GFSK": 1, "AM": 0, "FM": 0}[m]


def map_bits(mod: str, bits: np.ndarray) -> np.ndarray:
    """Gray-coded bit -> symbol mapping.  `bits` length must be a multiple of k."""
    m = mod.upper()
    k = bits_per_symbol(m)
    b = np.asarray(bits, dtype=np.uint8).ravel()
    n_sym = b.size // k
    b = b[: n_sym * k].reshape(n_sym, k)
    c = constellation(m)
    idx = np.zeros(n_sym, dtype=np.int64)
    for j in range(k):
        idx = (idx << 1) | b[:, j]
    return c[idx]


def demap_symbols(mod: str, symbols: np.ndarray) -> np.ndarray:
    """Nearest-point symbol -> Gray bit demapping (hard decisions)."""
    m = mod.upper()
    k = bits_per_symbol(m)
    c = constellation(m)
    s = np.asarray(symbols, dtype=np.complex128).ravel()
    if s.size == 0:
        return np.zeros(0, dtype=np.uint8)
    idx = np.argmin(np.abs(s[:, None] - c[None, :]) ** 2, axis=1)
    bits = np.empty((idx.size, k), dtype=np.uint8)
    for j in range(k):
        bits[:, j] = (idx >> (k - 1 - j)) & 1
    return bits.ravel()


# --------------------------------------------------------------------------- #
#  Pulse shaping / modulators
# --------------------------------------------------------------------------- #
def rrc_taps(beta: float, sps: float, span: int = 10) -> np.ndarray:
    """Root-raised-cosine FIR taps (unit energy).  `sps` may be fractional."""
    beta = float(np.clip(beta, 1e-4, 0.999))
    sps = float(sps)
    half = int(math.ceil(span * sps))
    n = np.arange(-half, half + 1, dtype=np.float64)
    t = n / sps
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(np.pi * t * (1 + beta))
        den = np.pi * t * (1 - (4 * beta * t) ** 2)
        h = num / den
    h[np.isnan(h)] = 1.0
    idx0 = int(np.argmin(np.abs(t)))
    h[idx0] = 1 - beta + 4 * beta / np.pi
    if span > 0:
        h[np.abs(t) == 1 / (4 * beta)] = (beta / math.sqrt(2)) * (
            (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
        )
    return h / math.sqrt(float(np.sum(h ** 2)))


def pulse_shape(symbols: np.ndarray, sps: int, beta: float, span: int = 10, offset: float = 0.0) -> np.ndarray:
    """Upsample symbols to `sps` and apply the RRC matched pulse."""
    h = rrc_taps(beta, sps, span)
    x = np.zeros(symbols.size * sps, dtype=np.complex128)
    x[::sps] = symbols
    y = sigproc.upfirdn(h, x)
    delay = (h.size - 1) // 2
    y = y[delay: delay + symbols.size * sps]
    if y.size < symbols.size * sps:
        y = np.pad(y, (0, symbols.size * sps - y.size))
    if offset:
        y = np.roll(y, int(round(offset * sps)))
    return y




def _gaussian_freq_pulse(sps: int, bt: float, span: int = 3) -> np.ndarray:
    """Gaussian-filtered NRZ frequency pulse for GFSK (BT product)."""
    if bt <= 0:
        return np.ones(sps) / sps
    n = np.arange(-span * sps / 2, span * sps / 2 + 1)
    t = n / sps
    alpha = math.sqrt(math.log(2) / 2) / bt
    g = (np.sqrt(2 * math.pi) / alpha) * np.exp(-2 * (math.pi ** 2) * (t ** 2) / (alpha ** 2))
    g /= g.sum()
    return g


def modulate_cpfsk(bits: np.ndarray, sps: int, h: float = 0.5, bt: float = 0.0,
                   n_levels: int = 2) -> np.ndarray:
    """Continuous-phase FSK / GFSK.  `h` = 2*f_dev*Ts (modulation index)."""
    b = np.asarray(bits, dtype=np.int64).ravel()
    if n_levels > 2:
        k = int(round(math.log2(n_levels)))
        n_sym = b.size // k
        b = (b[: n_sym * k].reshape(n_sym, k) * (2 ** np.arange(k - 1, -1, -1))).sum(axis=1)
        levels = 2.0 * b / (n_levels - 1) - 1.0
    else:
        levels = 2.0 * b - 1.0
    freq = np.repeat(levels, sps)
    if bt > 0:
        g = _gaussian_freq_pulse(sps, bt)
        freq = np.convolve(freq, g, mode="same")
    phase = np.cumsum(2 * np.pi * h * freq / (2.0 * sps))
    return np.exp(1j * phase)


def modulate_am(fs: float, message: np.ndarray, message_fs: float, carrier_hz: float,
                mod_index: float = 0.7) -> np.ndarray:
    """Real passband AM-DSB (large carrier)."""
    n = int(round(message.size * fs / message_fs))
    msg = np.interp(np.linspace(0, message.size, n, endpoint=False), np.arange(message.size), message)
    msg = msg / max(np.max(np.abs(msg)), 1e-9)
    t = np.arange(n) / fs
    return (1.0 + mod_index * msg) * np.cos(2 * np.pi * carrier_hz * t) / (1.0 + mod_index / 2)


def modulate_fm(fs: float, message: np.ndarray, message_fs: float, carrier_hz: float,
                deviation_hz: float) -> np.ndarray:
    """Real passband FM."""
    n = int(round(message.size * fs / message_fs))
    msg = np.interp(np.linspace(0, message.size, n, endpoint=False), np.arange(message.size), message)
    msg = msg / max(np.max(np.abs(msg)), 1e-9)
    phase = 2 * np.pi * deviation_hz * np.cumsum(msg) / fs
    t = np.arange(n) / fs
    return np.cos(2 * np.pi * carrier_hz * t + phase)


def fm_broadcast_mpx(fs: float, audio_l: np.ndarray, audio_r: np.ndarray, audio_fs: float,
                     deviation_hz: float = 75000.0) -> np.ndarray:
    """Stereo FM broadcast multiplex (complex baseband), standard MPX layout.

    MPX = 0.9*(L+R) + 0.9*(L-R)*cos(2*pi*38k*t) + 0.09*cos(2*pi*19k*t) [+ RDS]
    """
    n = int(round(audio_l.size * fs / audio_fs))
    def rs(x):
        s = np.interp(np.linspace(0, x.size, n, endpoint=False), np.arange(x.size), x)
        return s / max(np.max(np.abs(s)), 1e-9)
    l, r = rs(audio_l), rs(audio_r)
    t = np.arange(n) / fs
    mpx = 0.9 * (l + r) + 0.9 * (l - r) * np.cos(2 * np.pi * 38000 * t) + 0.09 * np.cos(2 * np.pi * 19000 * t)
    phase = 2 * np.pi * deviation_hz * np.cumsum(mpx) / fs
    return np.exp(1j * phase)


# --------------------------------------------------------------------------- #
#  FEC + interleaving (transmit side)
# --------------------------------------------------------------------------- #
@dataclass
class ConvCode:
    k: int = 1
    n: int = 2
    K: int = 7
    polys: tuple = (0o171, 0o133)

    @property
    def rate(self) -> float:
        return self.k / self.n

    def to_dict(self) -> dict:
        return {"k": self.k, "n": self.n, "K": self.K,
                "polys": [int(p) for p in self.polys], "rate": self.rate}


STANDARD_CONV_CODES = [
    ConvCode(1, 2, 3, (0o7, 0o5)),
    ConvCode(1, 2, 5, (0o23, 0o35)),
    ConvCode(1, 2, 7, (0o171, 0o133)),   # CCSDS / Voyager / 802.11
    ConvCode(1, 3, 7, (0o171, 0o133, 0o165)),
    ConvCode(1, 2, 9, (0o561, 0o753)),
    ConvCode(1, 3, 9, (0o561, 0o753, 0o711)),
]


def conv_encode(bits: np.ndarray, code: ConvCode) -> np.ndarray:
    """Rate-1/n convolutional encoder (tail-terminated) - vectorised over states."""
    b = np.asarray(bits, dtype=np.uint8).ravel()
    K = code.K
    mask = (1 << (K - 1)) - 1
    state = 0
    out_bits = np.empty(b.size * code.n, dtype=np.uint8)
    for i, bit in enumerate(b):
        reg = ((state << 1) | int(bit)) & ((1 << K) - 1)
        for j, p in enumerate(code.polys):
            out_bits[i * code.n + j] = bin(reg & p).count("1") & 1
        state = reg & mask
    # flush with zeros
    for _ in range(K - 1):
        reg = (state << 1) & ((1 << K) - 1)
        for j, p in enumerate(code.polys):
            out_bits = np.append(out_bits, bin(reg & p).count("1") & 1)
        state = reg & mask
    return out_bits.astype(np.uint8)


# --------------------------------------------------------------------------- #
#  Interleavers (forward transforms)
# --------------------------------------------------------------------------- #
def interleave_indices(n: int, kind: str, **kw) -> np.ndarray:
    """Permutation applied by :func:`interleave`: ``interleave(b) == b[perm]``.

    Supported kinds (matching the platform's interleaver catalogue):

    ``block``        write rows, read columns (repeating row-column interleaver with a
                     ``rows x rows`` block; standard telemetry construction)
    ``convolutional`` streaming Forney interleaver: branch ``i`` is delayed by ``i * depth``
                     symbols, so it is defined on the stream and not on a fixed block
    ``diagonal``     write rows, read diagonals (repeating, same block as ``block``)
    ``pseudo-random``  a seeded random permutation
    """
    n = int(n)
    kind = (kind or "none").lower()
    if kind in ("none", "", "off"):
        return np.arange(n, dtype=np.int64)
    if kind == "pseudo-random":
        # Block-random interleaver: a seeded random permutation applied inside each block of
        # ``rows x rows`` bits.  Also made repeating (see below) so that a receiver can undo it from
        # a stream whose start alignment is unknown - a single random permutation of the whole record
        # would be unrecoverable from any truncated capture.
        seed = int(kw.get("seed", 12345))
        rows = int(max(2, int(kw.get("rows", 16))))
        block_len = rows * rows
        n_full = (n // block_len) * block_len
        perm = np.arange(n, dtype=np.int64)
        rng = np.random.default_rng(seed)
        order = rng.permutation(block_len)
        for start in range(0, n_full, block_len):
            idx = np.arange(start, start + block_len, dtype=np.int64)
            perm[idx[order]] = idx
        return perm
    if kind == "convolutional":
        rows = int(kw.get("rows", 16))
        depth = int(kw.get("depth", 4))
        i = np.arange(n, dtype=np.int64)
        # output position of input symbol i: i + (i mod rows) * depth
        pos = i + (i % max(1, rows)) * max(1, depth)
        perm = np.argsort(np.argsort(pos))
        return perm
    rows = int(max(2, int(kw.get("rows", 16))))
    # The row-column and diagonal interleavers are *repeating*: the record is processed in blocks of
    # ``block_len = rows x rows`` bits (the last partial block is passed through unchanged).  A
    # repeating construction is what a receiver can undo, because it does not need to know where the
    # transmitter's record started - only the offset within one block.  A whole-record transpose (the
    # earlier behaviour) is unrecoverable from any stream that lost its first symbols.
    block_len = rows * rows
    n_full = (n // block_len) * block_len
    perm = np.arange(n, dtype=np.int64)
    if n_full == 0:
        return perm
    for start in range(0, n_full, block_len):
        idx = np.arange(start, start + block_len, dtype=np.int64).reshape(rows, rows)
        if kind == "block":
            fwd = idx.T.ravel()
        elif kind == "diagonal":
            order = []
            for d in range(2 * rows - 1):
                for r in range(rows):
                    c = d - r
                    if 0 <= c < rows:
                        order.append(idx[r, c])
            fwd = np.array(order, dtype=np.int64)
        else:
            raise ValueError(f"unknown interleaver kind: {kind}")
        # fwd[k] is the source position of output k within the stream: build the inverse map
        perm[fwd] = idx.ravel()
    return perm


def interleave(bits: np.ndarray, kind: str, **kw) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8).ravel()
    return b[interleave_indices(b.size, kind, **kw)].astype(np.uint8)


def interleave_deindices(n: int, kind: str, **kw) -> np.ndarray:
    """Permutation that inverts :func:`interleave` (``b == interleave(b)[deindices]``)."""
    perm = interleave_indices(n, kind, **kw)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size, dtype=perm.dtype)
    return inv


def deinterleave(bits: np.ndarray, kind: str, **kw) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8).ravel()
    return b[interleave_deindices(b.size, kind, **kw)].astype(np.uint8)


def deinterleave_factory(kind: str, **kw):
    """Convenience: return ``f(bits) -> deinterleaved bits`` for a given geometry."""
    def _apply(bits: np.ndarray) -> np.ndarray:
        return deinterleave(bits, kind, **kw)
    return _apply


INTERLEAVER_CANDIDATES = [
    ("none", {}),
    ("block", {"rows": 8}), ("block", {"rows": 16}), ("block", {"rows": 32}),
    ("convolutional", {"rows": 8, "depth": 2}), ("convolutional", {"rows": 16, "depth": 4}),
    ("diagonal", {"rows": 8}), ("diagonal", {"rows": 16}), ("diagonal", {"rows": 32}),
    ("pseudo-random", {"seed": 12345}), ("pseudo-random", {"seed": 999}),
]


def add_awgn(x: np.ndarray, snr_db: float, fs: float, bandwidth_hz: float,
             complex_signal: bool = True, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add white Gaussian noise so the *in-band* SNR equals `snr_db`.

    N0 is derived from the signal power and its occupied bandwidth, then scaled
    to the sampling bandwidth, i.e. wide captures are noisier - exactly how an
    SDR front-end behaves.
    """
    rng = rng or np.random.default_rng(0)
    p_sig = float(np.mean(np.abs(x) ** 2))
    if p_sig <= 0 or not np.isfinite(snr_db):
        return x.copy()
    snr_lin = 10 ** (snr_db / 10.0)
    bw = max(bandwidth_hz, fs / 1000.0)
    n0 = p_sig / (snr_lin * bw)                 # noise power per Hz
    p_noise = n0 * fs                           # total noise power over fs
    if complex_signal:
        sigma = math.sqrt(p_noise / 2.0)
        n = sigma * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
    else:
        sigma = math.sqrt(p_noise)
        n = sigma * rng.standard_normal(x.size)
    return x + n


def apply_burst_errors(symbols: np.ndarray, burst_len: int, period: int, amplitude: float = 1.6,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """Corrupt short bursts of symbols (models impulsive interference / fading)."""
    rng = rng or np.random.default_rng(1)
    y = symbols.copy()
    if burst_len <= 0 or period <= 0:
        return y
    for start in range(0, symbols.size, period):
        end = min(start + burst_len, symbols.size)
        ph = rng.uniform(0, 2 * np.pi)
        y[start:end] += amplitude * np.exp(1j * ph)
    return y


def apply_iq_imbalance(x: np.ndarray, gain_db: float = 0.0, phase_deg: float = 0.0) -> np.ndarray:
    g = 10 ** (gain_db / 20.0)
    p = math.radians(phase_deg)
    i, q = x.real, x.imag
    return (i * g + 1j * (q * math.cos(p) + i * math.sin(p))).astype(np.complex128)


# --------------------------------------------------------------------------- #
#  Top level generation
# --------------------------------------------------------------------------- #
DEFAULT_SPEC: dict[str, Any] = {
    "modulation": "QPSK",
    "fs": 200000.0,
    "symbol_rate": 25000.0,
    "snr_db": 18.0,
    "n_symbols": 4000,
    "rolloff": 0.35,
    "carrier_offset_hz": 5000.0,
    "phase_offset_deg": 0.0,
    "payload": "random",             # random | text
    "text": "SIH26147 RF ANALYSIS PLATFORM DEMO PAYLOAD ",
    "prefix_hex": "",                # known sync word placed once at the start
    "fec": "none",                   # none | conv_K7_r1_2 | conv_K5_r1_2 | conv_K3_r1_2 | conv_K7_r1_3 | rs_255_223
    "interleaver": "none",           # none | block | convolutional | diagonal | pseudo-random
    "interleaver_rows": 16,
    "interleaver_depth": 4,
    "interleaver_seed": 12345,
    "burst_len": 0,
    "burst_period": 0,
    "iq_gain_db": 0.0,
    "iq_phase_deg": 0.0,
    "dc_offset": 0.0,
    "fs_deviation_hz": None,         # FSK
    "mod_index": 0.5,                # FSK h
    "bt": 0.35,                      # GFSK BT
    "am_index": 0.7,
    "fm_deviation_hz": 15000.0,
    "tone_hz": 1200.0,               # AM/FM test message
    "scramble": True,
    "clipping": 0.0,
}

FEC_PRESETS = {
    "none": None,
    "conv_K3_r1_2": ConvCode(1, 2, 3, (0o7, 0o5)),
    "conv_K5_r1_2": ConvCode(1, 2, 5, (0o23, 0o35)),
    "conv_K7_r1_2": ConvCode(1, 2, 7, (0o171, 0o133)),
    "conv_K9_r1_2": ConvCode(1, 2, 9, (0o561, 0o753)),
    "conv_K7_r1_3": ConvCode(1, 3, 7, (0o171, 0o133, 0o165)),
}


@dataclass
class GeneratedSignal:
    samples: np.ndarray
    fs: float
    is_complex: bool
    ground_truth: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("samples")
        return d


def multiplicative_scramble(bits: np.ndarray) -> np.ndarray:
    """Self-synchronising scrambler (1 + x^9 + x^11) applied by the generator before modulation.

    ``out[i] = bits[i] XOR out[i-9] XOR out[i-11]`` - the shift register is driven by the *output*,
    which is what makes a receiver able to synchronise on the stream itself.  The inverse is
    :func:`multiplicative_descramble` (identical polynomial, register driven by the input).
    """
    b = np.asarray(bits, dtype=np.uint8).ravel().copy()
    reg = np.zeros(11, dtype=np.uint8)
    out = np.empty_like(b)
    for i in range(b.size):
        fb = reg[8] ^ reg[10]
        bit = b[i] ^ fb
        reg = np.roll(reg, 1)
        reg[0] = bit
        out[i] = bit
    return out


def multiplicative_descramble(bits: np.ndarray) -> np.ndarray:
    """Exact inverse of :func:`multiplicative_scramble` - recovers the payload bit order.

    ``out[i] = bits[i] XOR bits[i-9] XOR bits[i-11]`` with the register driven by the *input*, so
    ``multiplicative_descramble(multiplicative_scramble(b)) == b`` (verified in the tests).  A
    receiver that has recovered the payload bits applies this once before reading ASCII text.
    """
    b = np.asarray(bits, dtype=np.uint8).ravel()
    out = b.copy()
    for i in range(b.size):
        fb = (b[i - 9] if i >= 9 else 0) ^ (b[i - 11] if i >= 11 else 0)
        out[i] = b[i] ^ fb
    return out


def _payload_bits(n_bits: int, spec: dict, rng: np.random.Generator) -> np.ndarray:
    prefix = bytes.fromhex(spec.get("prefix_hex") or "")
    if spec.get("payload", "random") == "text":
        text = spec.get("text") or "SIH26147 DEMO PAYLOAD "
        data = (prefix + (text.encode("ascii") * (n_bits // 8 // len(text) + 2)))
        bits = bsu.bytes_to_bits(data)
    else:
        bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
        if prefix:
            pre = bsu.bytes_to_bits(prefix)
            bits = np.concatenate([pre, bits])
    if spec.get("scramble", True):
        # multiplicative scrambler 1+x^9+x^11 (as used before modulation in many links)
        bits = multiplicative_scramble(bits)
    return bits.astype(np.uint8)


def generate_signal(spec: dict | None = None) -> GeneratedSignal:
    cfg = dict(DEFAULT_SPEC)
    cfg.update(spec or {})
    rng = np.random.default_rng(int(cfg.get("seed", 20260926)))
    mod = str(cfg["modulation"]).upper()
    fs = float(cfg["fs"])
    rs = float(cfg["symbol_rate"])
    warnings: list[str] = []

    gt: dict[str, Any] = {
        "modulation": mod, "fs": fs, "symbol_rate": rs, "snr_db": cfg["snr_db"],
        "carrier_offset_hz": cfg["carrier_offset_hz"], "rolloff": cfg["rolloff"],
        "fec": cfg["fec"], "interleaver": cfg["interleaver"], "generator_seed": int(cfg.get("seed", 20260926)),
    }

    sps_i = int(max(2, round(fs / max(rs, 1e-9))))
    rs_actual = fs / sps_i
    gt["samples_per_symbol"] = sps_i
    gt["symbol_rate_requested"] = rs
    if abs(rs_actual - rs) / max(rs, 1e-9) > 1e-6:
        warnings.append(
            f"symbol rate quantised to {rs_actual:g} sym/s: with fs={fs:g} Hz the generator uses "
            f"{sps_i} samples/symbol (a non-integer samples-per-symbol ratio would require a resampler "
            f"and is avoided so the synthetic data stays exactly reproducible)")
        gt["symbol_rate_quantised"] = True
    rs = rs_actual
    gt["symbol_rate"] = rs

    # ---------------- real passband modulations (AM / FM) ----------------
    if mod in ("AM", "FM"):
        dur = cfg.get("duration_s", 0.25)
        n = int(fs * dur)
        msg_fs = 48000.0
        t_msg = np.arange(int(msg_fs * dur)) / msg_fs
        msg = (np.sin(2 * np.pi * cfg["tone_hz"] * t_msg)
               + 0.4 * np.sin(2 * np.pi * 3 * cfg["tone_hz"] * t_msg)
               + 0.2 * np.sin(2 * np.pi * 7 * cfg["tone_hz"] * t_msg))
        carrier = float(cfg.get("carrier_hz", fs * 0.2))
        if mod == "AM":
            x = modulate_am(fs, msg, msg_fs, carrier, float(cfg["am_index"]))
        else:
            x = modulate_fm(fs, msg, msg_fs, carrier, float(cfg["fm_deviation_hz"]))
        gt.update({"carrier_hz": carrier, "duration_s": dur, "tone_hz": cfg["tone_hz"]})
        x = add_awgn(x, float(cfg["snr_db"]), fs, 2 * (cfg["fm_deviation_hz"] if mod == "FM" else cfg["tone_hz"] * 3),
                     complex_signal=False, rng=rng)
        if cfg.get("clipping"):
            a = float(cfg["clipping"])
            x = np.clip(x, -a, a)
        x = x / max(np.max(np.abs(x)), 1e-9) * 0.9
        return GeneratedSignal(x.astype(np.float64), fs, False, gt, warnings)

    if mod == "FM_STEREO":
        dur = float(cfg.get("duration_s", 0.4))
        n_audio = int(48000 * dur)
        t = np.arange(n_audio) / 48000.0
        l = 0.6 * np.sin(2 * np.pi * 440 * t) * (1 + 0.3 * np.sin(2 * np.pi * 2 * t))
        r = 0.6 * np.sin(2 * np.pi * 660 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
        x = fm_broadcast_mpx(fs, l, r, 48000.0, float(cfg["fm_deviation_hz"]))
        gt.update({"carrier_hz": 0.0, "duration_s": dur, "audio": "stereo 440/660 Hz test tones",
                   "deviation_hz": cfg["fm_deviation_hz"], "bandwidth_hz": 2 * cfg["fm_deviation_hz"]})
        x = add_awgn(x, float(cfg["snr_db"]), fs, 2 * cfg["fm_deviation_hz"], complex_signal=True, rng=rng)
        p = math.sqrt(float(np.mean(np.abs(x) ** 2)))
        return GeneratedSignal((x / p * 0.9).astype(np.complex64), fs, True, gt, warnings)

    # ---------------- complex baseband digital modulations ----------------
    n_sym = int(cfg["n_symbols"])
    code: ConvCode | None = FEC_PRESETS.get(str(cfg["fec"]), None)
    k = bits_per_symbol(mod) if mod not in ("2FSK", "GFSK") else 1
    n_payload_bits = n_sym * k
    if code is not None:
        n_payload_bits = max(64, int(n_payload_bits * code.k / code.n) - (code.K - 1) * code.k)
    bits_tx = _payload_bits(n_payload_bits, cfg, rng)
    gt["payload_bits"] = int(bits_tx.size)
    # the transmitted bit string is kept in the ground truth so a receiver's recovered bits can be
    # compared with what was actually sent (message-recovery verification, not a claim)
    gt["payload_bitstring"] = "".join(str(int(b)) for b in bits_tx[:8192])
    gt["payload_bitstring_truncated"] = bool(bits_tx.size > 8192)
    n_bytes = int(bits_tx.size) // 8
    gt["payload_bytes_hex"] = np.packbits(bits_tx[:n_bytes * 8].astype(np.uint8)).tobytes().hex()
    if cfg.get("payload") == "text":
        gt["payload_text"] = str(cfg.get("text"))
        try:
            gt["payload_text_bytes"] = len(str(cfg.get("text")).encode("utf-8"))
        except Exception:
            gt["payload_text_bytes"] = None
    gt["payload_text"] = None
    if cfg.get("payload") == "text":
        gt["payload_text"] = str(cfg.get("text"))

    bits_coded = conv_encode(bits_tx, code) if code is not None else bits_tx
    if code is not None:
        gt["fec_code"] = code.to_dict()
    bits_int = interleave(bits_coded, cfg["interleaver"], rows=cfg["interleaver_rows"],
                          depth=cfg["interleaver_depth"], seed=cfg["interleaver_seed"])
    gt["n_coded_bits"] = int(bits_int.size)

    if mod in ("2FSK", "GFSK"):
        sps_f = max(2, int(round(fs / rs)))
        gt["symbol_rate"] = fs / sps_f
        gt["samples_per_symbol"] = sps_f
        x = modulate_cpfsk(bits_int, sps_f, h=float(cfg["mod_index"]),
                           bt=float(cfg["bt"]) if mod == "GFSK" else 0.0, n_levels=2)
        gt["fsk_mod_index"] = cfg["mod_index"]
        gt["fsk_deviation_hz"] = float(cfg["mod_index"]) * gt["symbol_rate"] / 2.0
        gt["bandwidth_hz"] = (1 + float(cfg["mod_index"])) * gt["symbol_rate"]
    else:
        syms = map_bits(mod, bits_int)
        gt["n_symbols"] = int(syms.size)
        if cfg.get("burst_len"):
            syms = apply_burst_errors(syms, int(cfg["burst_len"]), int(cfg["burst_period"]), rng=rng)
            gt["burst_errors"] = {"len_symbols": cfg["burst_len"], "period_symbols": cfg["burst_period"]}
        x = pulse_shape(syms, sps_i, float(cfg["rolloff"]), span=int(cfg.get("span", 8)))
        gt["bandwidth_hz"] = rs * (1 + float(cfg["rolloff"]))

    # carrier / phase offset
    t = np.arange(x.size) / fs
    f_off = float(cfg["carrier_offset_hz"])
    ph = math.radians(float(cfg["phase_offset_deg"]))
    x = x * np.exp(1j * (2 * np.pi * f_off * t + ph))

    if float(cfg.get("iq_gain_db", 0.0)) or float(cfg.get("iq_phase_deg", 0.0)):
        x = apply_iq_imbalance(x, float(cfg["iq_gain_db"]), float(cfg["iq_phase_deg"]))
        gt["iq_imbalance"] = {"gain_db": cfg["iq_gain_db"], "phase_deg": cfg["iq_phase_deg"]}
    if float(cfg.get("dc_offset", 0.0)):
        x = x + float(cfg["dc_offset"]) * (1 + 1j)

    bw = float(gt.get("bandwidth_hz", rs))
    if cfg.get("snr_db") is None:
        gt["snr_db"] = None
        gt["noise"] = "no noise added (snr_db = null)"
    else:
        # the SNR is defined in-band, i.e. over the emission's own occupied bandwidth
        x = add_awgn(x, float(cfg["snr_db"]), fs, bw, complex_signal=True, rng=rng)
    peak = float(np.max(np.abs(x))) or 1.0
    x = (x / peak) * 0.9
    if cfg.get("clipping"):
        a = float(cfg["clipping"])
        x = np.clip(x.real, -a, a) + 1j * np.clip(x.imag, -a, a)
        warnings.append("clipping applied (non-linear distortion)")
    gt["duration_s"] = float(x.size / fs)
    gt["peak_amplitude"] = float(np.max(np.abs(x)))
    return GeneratedSignal(x.astype(np.complex64), fs, True, gt, warnings)


# --------------------------------------------------------------------------- #
#  File writers (IQ + WAV)
# --------------------------------------------------------------------------- #
IQ_FORMATS = {
    "int8": (np.int8, 127.0),
    "uint8": (np.uint8, 127.5),
    "int16": (np.int16, 32767.0),
    "uint16": (np.uint16, 32767.5),
    "int32": (np.int32, 2147483647.0),
    "float32": (np.float32, 1.0),
    "float64": (np.float64, 1.0),
}


def write_iq(path: str, samples: np.ndarray, fmt: str = "int16", endian: str = "<",
             interleaved: bool = True) -> str:
    """Write complex samples as raw interleaved (or split) IQ binary."""
    dtype, scale = IQ_FORMATS[fmt]
    x = np.asarray(samples)
    i = x.real
    q = x.imag
    if np.dtype(dtype).kind == "f":
        arr = np.empty(2 * i.size, dtype=dtype)
        if interleaved:
            arr[0::2] = i.astype(dtype)
            arr[1::2] = q.astype(dtype)
        else:
            arr[: i.size] = i.astype(dtype)
            arr[i.size:] = q.astype(dtype)
        arr.astype(f"{endian}{np.dtype(dtype).str[1:]}").tofile(path)
        return path
    if fmt in ("uint8", "uint16"):
        idx = np.clip(i * scale + (128 if fmt == "uint8" else 32768), 0, np.iinfo(dtype).max)
        qdx = np.clip(q * scale + (128 if fmt == "uint8" else 32768), 0, np.iinfo(dtype).max)
    else:
        idx = np.clip(np.round(i * scale), np.iinfo(dtype).min, np.iinfo(dtype).max)
        qdx = np.clip(np.round(q * scale), np.iinfo(dtype).min, np.iinfo(dtype).max)
    arr = np.empty(2 * i.size, dtype=dtype)
    if interleaved:
        arr[0::2] = idx.astype(dtype)
        arr[1::2] = qdx.astype(dtype)
    else:
        arr[: i.size] = idx.astype(dtype)
        arr[i.size:] = qdx.astype(dtype)
    arr.astype(f"{endian}{np.dtype(dtype).str[1:]}").tofile(path)
    return path


def write_wav_iq(path: str, samples: np.ndarray, fs: float, fmt: str = "int16") -> str:
    """Write complex baseband as a 2-channel WAV (L=I, R=Q)."""
    x = np.asarray(samples)
    if x.ndim == 1:
        x = x.astype(np.complex128)
    if fmt == "float32":
        data = np.stack([x.real, x.imag], axis=1).astype(np.float32)
    else:
        data = np.stack([np.clip(x.real, -1, 1), np.clip(x.imag, -1, 1)], axis=1)
        data = (data * 32767.0).astype(np.int16)
    from scipy.io import wavfile
    wavfile.write(path, int(round(fs)), data)
    return path


def write_wav_real(path: str, samples: np.ndarray, fs: float) -> str:
    from scipy.io import wavfile
    x = np.asarray(samples, dtype=np.float64).ravel()
    peak = float(np.max(np.abs(x))) or 1.0
    wavfile.write(path, int(round(fs)), (np.clip(x / peak, -1, 1) * 32767).astype(np.int16))
    return path
