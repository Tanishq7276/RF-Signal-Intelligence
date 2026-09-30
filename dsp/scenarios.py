"""Standard, reproducible test scenarios.

The same builders are used by the automated tests, by ``Benchmark Mode`` in the UI and by the
demo-data generator, so that what the platform claims to measure can be checked against the exact
ground truth the platform itself synthesised.

Everything here is deterministic: a fixed seed per scenario.
"""

from __future__ import annotations

import os

import numpy as np

from . import synth

__all__ = ["noiseless", "scale_to_snr", "signal_scenario", "multi_signal_mix", "gated_burst",
           "noise_only", "iq_imbalance_case", "SCENARIOS", "build_scenario"]


def noiseless(**spec) -> tuple[np.ndarray, float, dict]:
    """Generate a signal without noise (the generator's own ``snr_db=None`` path)."""
    spec = dict(spec)
    spec["snr_db"] = None
    g = synth.generate_signal(spec)
    return g.samples, g.fs, g.ground_truth


def scale_to_snr(x: np.ndarray, fs: float, snr_db: float, n0: float,
                 bandwidth_hz: float) -> np.ndarray:
    """Scale ``x`` so that its in-band SNR against a noise density ``n0`` equals ``snr_db``."""
    p = float(np.mean(np.abs(x) ** 2))
    if p <= 0:
        return x
    target_p = (10 ** (snr_db / 10.0)) * n0 * max(bandwidth_hz, 1.0)
    return x * np.sqrt(target_p / p)


def _noise(n: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad a generated signal to exactly ``n`` samples."""
    if x.size >= n:
        return x[:n]
    return np.pad(x, (0, n - x.size))


def _nsym(n: int, fs: float, rs: float, extra: int = 24) -> int:
    """Number of symbols needed to fill ``n`` samples at ``rs``."""
    return int(np.ceil(n * rs / fs)) + extra


def multi_signal_mix(which: str = "A", seed: int | None = None) -> dict:
    """Two/three-carrier captures used to validate the multi-signal detector.

    Returns ``samples``, ``fs``, ``truth`` (list of per-emission truth records) and the complex
    noise density of the record.
    """
    if which.upper() == "A":
        seed = 11 if seed is None else seed
        rng = np.random.default_rng(seed)
        n = 40000
        fs = 200000.0
        t = np.arange(n) / fs
        qpsk, _, _ = noiseless(modulation="QPSK", symbol_rate=25000.0, fs=fs,
                               n_symbols=_nsym(n, fs, 25000.0), rolloff=0.35,
                               carrier_offset_hz=-30000.0, seed=101)
        bpsk, _, _ = noiseless(modulation="BPSK", symbol_rate=12000.0, fs=fs,
                               n_symbols=_nsym(n, fs, 12000.0), rolloff=0.35,
                               carrier_offset_hz=25000.0, seed=102)
        cw = 0.05 * np.exp(2j * np.pi * 60000.0 * t)
        qpsk = _fit(qpsk, n)
        bpsk = _fit(bpsk, n)
        sigma = 0.004
        n0 = 2.0 * sigma ** 2 / fs
        qpsk = scale_to_snr(qpsk, fs, 18.0, n0, 25000.0 * 1.35)
        bpsk = scale_to_snr(bpsk, fs, 12.0, n0, 12000.0 * 1.35)
        x = qpsk + bpsk + cw + _noise(n, sigma, rng)
        truth = [
            {"modulation": "QPSK", "symbol_rate": 25000.0, "carrier_offset_hz": -30000.0,
             "snr_db": 18.0, "kind": "modulated"},
            {"modulation": "BPSK", "symbol_rate": 12000.0, "carrier_offset_hz": 25000.0,
             "snr_db": 12.0, "kind": "modulated"},
            {"modulation": "CW", "symbol_rate": None, "carrier_offset_hz": 60000.0,
             "snr_db": None, "amplitude": 0.05, "kind": "tone"},
        ]
        return {"name": "mix_A", "samples": x.astype(np.complex64), "fs": fs, "truth": truth,
                "noise_density": n0, "sigma": sigma, "n": n,
                "description": "QPSK (-30 kHz, 18 dB) + BPSK (+25 kHz, 12 dB) + CW tone "
                               "(+60 kHz) + AWGN"}

    seed = 12 if seed is None else seed
    rng = np.random.default_rng(seed)
    n = 60000
    fs = 200000.0
    t = np.arange(n) / fs
    sigma = 0.002
    n0 = 2.0 * sigma ** 2 / fs
    specs = [("QPSK", 15000.0, 0.0, 25.0, 201), ("QPSK", 15000.0, 50000.0, 10.0, 202),
             ("QPSK", 15000.0, -60000.0, 8.0, 203)]
    x = np.zeros(n, dtype=np.complex128)
    truth = []
    for mod, rs, off, snr, sd in specs:
        s, _, _ = noiseless(modulation=mod, symbol_rate=rs, fs=fs,
                            n_symbols=_nsym(n, fs, rs), rolloff=0.35,
                            carrier_offset_hz=off, seed=sd)
        s = _fit(s, n)
        x = x + scale_to_snr(s, fs, snr, n0, rs * 1.35)
        truth.append({"modulation": mod, "symbol_rate": rs, "carrier_offset_hz": off,
                      "snr_db": snr, "kind": "modulated"})
    x = x + _noise(x.size, sigma, rng)
    return {"name": "mix_B", "samples": x.astype(np.complex64), "fs": fs, "truth": truth,
            "noise_density": n0, "sigma": sigma, "n": x.size,
            "description": "three QPSK carriers at 25 / 10 / 8 dB in-band SNR "
                           "(0, +50, -60 kHz) + AWGN"}


def gated_burst(seed: int | None = None) -> dict:
    """BPSK that transmits 1/3.33 of the time, plus a continuous CW interferer."""
    seed = 13 if seed is None else seed
    rng = np.random.default_rng(seed)
    n = 120000
    fs = 200000.0
    t = np.arange(n) / fs
    sigma = 0.01
    n0 = 2.0 * sigma ** 2 / fs
    bpsk, _, _ = noiseless(modulation="BPSK", symbol_rate=20000.0, fs=fs,
                           n_symbols=_nsym(n, fs, 20000.0), rolloff=0.35,
                           carrier_offset_hz=20000.0, seed=301)
    bpsk = _fit(bpsk, n)
    bpsk = scale_to_snr(bpsk, fs, 20.0, n0, 20000.0 * 1.35)
    gate = np.zeros(n, dtype=np.float64)
    period = int(0.06 * fs)                    # 60 ms frame
    on = int(0.018 * fs)                       # 18 ms of transmission -> 30 % duty cycle
    for start in range(0, n, period):
        gate[start:min(n, start + on)] = 1.0
    cw = 0.05 * np.exp(2j * np.pi * (-40000.0) * t)
    x = bpsk * gate + cw + _noise(n, sigma, rng)
    truth = [
        {"modulation": "BPSK", "symbol_rate": 20000.0, "carrier_offset_hz": 20000.0,
         "snr_db": 20.0, "kind": "burst", "duty_cycle": on / period},
        {"modulation": "CW", "symbol_rate": None, "carrier_offset_hz": -40000.0,
         "snr_db": None, "amplitude": 0.05, "kind": "tone"},
    ]
    return {"name": "gated_bpsk_cw", "samples": x.astype(np.complex64), "fs": fs, "truth": truth,
            "noise_density": n0, "sigma": sigma, "n": n,
            "description": "gated BPSK (+20 kHz, 30 % duty) + CW (-40 kHz) + AWGN"}


def noise_only(n: int = 60000, fs: float = 200000.0, sigma: float = 0.02,
               seed: int | None = None) -> dict:
    rng = np.random.default_rng(7 if seed is None else seed)
    return {"name": "noise_only", "samples": _noise(n, sigma, rng).astype(np.complex64), "fs": fs,
            "truth": [], "noise_density": 2.0 * sigma ** 2 / fs, "sigma": sigma, "n": n,
            "description": "AWGN only - nothing must be reported as a signal"}


def iq_imbalance_case(seed: int | None = None) -> dict:
    """16-QAM with a known IQ gain/phase imbalance and a known DC offset."""
    seed = 14 if seed is None else seed
    g = synth.generate_signal({
        "modulation": "16QAM", "symbol_rate": 25000.0, "fs": 200000.0, "snr_db": 25.0,
        "carrier_offset_hz": 4000.0, "iq_gain_db": 1.2, "iq_phase_deg": 4.0, "dc_offset": 0.02,
        "n_symbols": 3000, "seed": seed})
    return {"name": "iq_imbalance", "samples": g.samples, "fs": g.fs, "truth": [
        {"modulation": "16QAM", "symbol_rate": 25000.0, "carrier_offset_hz": 4000.0,
         "snr_db": 25.0, "kind": "modulated"}],
        "noise_density": None, "sigma": None, "n": g.samples.size,
        "ground_truth": g.ground_truth,
        "description": "16-QAM with +1.2 dB IQ gain imbalance, 4 deg phase error and a DC offset"}


def signal_scenario(modulation: str, symbol_rate: float, snr_db: float, carrier_offset_hz: float,
                    fs: float = 200000.0, n_symbols: int = 4000, seed: int = 5, **extra) -> dict:
    """One modulated emission with a known SNR - the workhorse benchmark case."""
    g = synth.generate_signal({"modulation": modulation, "symbol_rate": symbol_rate, "fs": fs,
                               "snr_db": snr_db, "carrier_offset_hz": carrier_offset_hz,
                               "n_symbols": n_symbols, "seed": seed, **extra})
    gt = g.ground_truth
    return {"name": f"{modulation}_{int(symbol_rate)}sym_{int(snr_db)}dB", "samples": g.samples,
            "fs": fs, "truth": [{"modulation": modulation,
                                 "symbol_rate": gt.get("symbol_rate"),
                                 "carrier_offset_hz": carrier_offset_hz, "snr_db": snr_db,
                                 "kind": "modulated"}],
            "noise_density": None, "n": g.samples.size, "ground_truth": gt,
            "description": f"{modulation} at {symbol_rate:,.0f} sym/s, {snr_db:.0f} dB SNR, "
                           f"{carrier_offset_hz:+,.0f} Hz offset"}


def build_scenario(name: str) -> dict:
    """Look up a named scenario used by the benchmark UI."""
    table = {
        "mix_a": lambda: multi_signal_mix("A"),
        "mix_b": lambda: multi_signal_mix("B"),
        "gated": gated_burst,
        "noise_only": noise_only,
        "iq_imbalance": iq_imbalance_case,
    }
    if name in table:
        return table[name]()
    raise KeyError(f"unknown scenario '{name}'")


SCENARIOS = [
    {"id": "mix_a", "title": "Two carriers + CW tone",
     "description": "QPSK at -30 kHz (18 dB) and BPSK at +25 kHz (12 dB) plus an unmodulated "
                    "tone at +60 kHz in noise - tests multi-signal detection and per-emission "
                    "parameter extraction."},
    {"id": "mix_b", "title": "Three QPSK carriers (25 / 10 / 8 dB)",
     "description": "Three 15 kbaud QPSK emissions at 0, +50 and -60 kHz with decreasing SNR - "
                    "tests detection sensitivity and the ranking of emissions."},
    {"id": "gated", "title": "Gated burst + CW interferer",
     "description": "BPSK transmitting 30 % of the time (60 ms frame) plus a continuous tone - "
                    "tests burst segmentation and duty-cycle reporting."},
    {"id": "noise_only", "title": "Noise only (false-alarm test)",
     "description": "AWGN with no emission - the platform must report that nothing was found "
                    "instead of inventing a signal."},
    {"id": "iq_imbalance", "title": "16-QAM with IQ imbalance",
     "description": "16-QAM carrying +1.2 dB gain imbalance, 4 deg phase imbalance and a DC "
                    "offset - tests impairment measurement and correction."},
]


# ---------------------------------------------------------------------------------------
# shipped demo signals
# ---------------------------------------------------------------------------------------
#: The six recordings in ``sample_data/``.  They are generated with exactly these parameters,
#: so the file format detection, the parameter extractor and the classifier can all be checked
#: against a known ground truth (``sample_data/GROUND_TRUTH.json``).  The table is also used as a
#: fallback: if a demo file is missing from ``sample_data/`` the platform regenerates it instead of
#: showing a dead button.
DEMO_PRESETS: dict[str, dict] = {
    "sample_bpsk.wav": {
        "modulation": "BPSK", "symbol_rate": 25000.0, "fs": 200000.0, "snr_db": 22.0,
        "n_symbols": 6000, "carrier_offset_hz": 3500.0, "rolloff": 0.35, "fec": "none",
        "interleaver": "none", "seed": 101, "payload": "text",
        "text": "SIH26147 BPSK TELEMETRY FRAME 001",
        "description": "BPSK, 25 kbaud, 200 kS/s, 22 dB SNR, +3.5 kHz carrier offset, no FEC.",
    },
    "sample_qpsk.wav": {
        "modulation": "QPSK", "symbol_rate": 25000.0, "fs": 200000.0, "snr_db": 20.0,
        "n_symbols": 6000, "carrier_offset_hz": 4000.0, "rolloff": 0.35,
        "fec": "conv_K7_r1_2", "interleaver": "none", "seed": 102, "payload": "text",
        "text": "SIH26147 QPSK PAYLOAD CHANNEL A",
        "description": "QPSK, 25 kbaud, 200 kS/s, 20 dB SNR, +4 kHz offset, rate-1/2 K=7 "
                       "convolutional code - the FEC engine should find this one.",
    },
    "sample_8psk.wav": {
        "modulation": "8PSK", "symbol_rate": 25000.0, "fs": 200000.0, "snr_db": 26.0,
        "n_symbols": 6000, "carrier_offset_hz": -3000.0, "rolloff": 0.35, "fec": "none",
        "interleaver": "none", "seed": 103, "payload": "text",
        "text": "SIH26147 8PSK LINK TEST",
        "description": "8PSK, 25 kbaud, 26 dB SNR, -3 kHz carrier offset, uncoded.",
    },
    "sample_fsk.wav": {
        "modulation": "2FSK", "symbol_rate": 25000.0, "fs": 200000.0, "snr_db": 20.0,
        "n_symbols": 6000, "carrier_offset_hz": 2000.0, "mod_index": 0.5, "fec": "none",
        "interleaver": "none", "seed": 104, "payload": "text",
        "text": "SIH26147 FSK SENSOR BURST",
        "description": "2FSK, 25 kbaud (h = 0.5), 20 dB SNR, +2 kHz offset, uncoded.",
    },
    "sample_16qam.wav": {
        "modulation": "16QAM", "symbol_rate": 25000.0, "fs": 200000.0, "snr_db": 26.0,
        "n_symbols": 6000, "carrier_offset_hz": 3000.0, "rolloff": 0.30, "fec": "conv_K3_r1_2",
        "interleaver": "none", "seed": 105, "payload": "text",
        "text": "SIH26147 QAM BROADBAND BLOCK",
        "description": "16-QAM, 25 kbaud, 26 dB SNR, +3 kHz offset, rate-1/2 K=3 code.",
    },
    "sample_noisy_qpsk.wav": {
        "modulation": "QPSK", "symbol_rate": 25000.0, "fs": 200000.0, "snr_db": 6.0,
        "n_symbols": 6000, "carrier_offset_hz": 4000.0, "rolloff": 0.35,
        "fec": "conv_K7_r1_2", "interleaver": "none", "seed": 106, "payload": "text",
        "text": "SIH26147 LOW SNR CASE",
        "description": "QPSK at only 6 dB SNR - used to demonstrate that the platform reports low "
                       "confidence and uncertainty instead of inventing results.",
    },
}


def demo_spec(name: str) -> dict | None:
    """Spec of a shipped demo signal, or ``None`` when the name is unknown."""
    preset = DEMO_PRESETS.get(name) or DEMO_PRESETS.get(os.path.basename(name))
    if preset is None:
        return None
    spec = {k: v for k, v in preset.items() if k != "description"}
    spec["file_name"] = os.path.basename(name)
    return spec
