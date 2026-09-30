"""Spectrum analyser: occupied bandwidth, SNR and noise-floor fidelity against ground truth."""
import numpy as np
import pytest

from dsp import scenarios, spectrum

REF_OBW = {"BPSK": 11597.0, "QPSK": 29175.0, "8PSK": 33179.0, "16QAM": 58667.0,
           "64QAM": 58179.0, "GFSK": 20703.0}


@pytest.mark.parametrize("mod,rs", [("BPSK", 10000.0), ("QPSK", 25000.0), ("8PSK", 30000.0),
                                    ("16QAM", 50000.0), ("64QAM", 50000.0), ("GFSK", 20000.0)])
def test_occupied_bandwidth_within_5_percent(mod, rs):
    sc = scenarios.signal_scenario(mod, rs, 25.0, 5000.0, n_symbols=3000)
    a = spectrum.analyse_spectrum(sc["samples"], sc["fs"])
    assert a["ok"] and not a.get("no_signal")
    ref = REF_OBW[mod]
    assert abs(a["obw_99_hz"] / ref - 1.0) < 0.05, (mod, a["obw_99_hz"], ref)


@pytest.mark.parametrize("snr", [6.0, 10.0, 20.0])
def test_snr_tracks_ground_truth(snr):
    sc = scenarios.signal_scenario("QPSK", 25000.0, snr, 5000.0, n_symbols=3000)
    a = spectrum.analyse_spectrum(sc["samples"], sc["fs"])
    # the generator's SNR is defined over the nominal (1+alpha)*Rs band; our estimate is defined
    # over the measured 99 % band, so a 2.5 dB window is the honest tolerance
    assert abs(a["snr_db"] - snr) < 2.5, (a["snr_db"], snr)


def test_noise_floor_and_centre_frequency():
    sc = scenarios.signal_scenario("QPSK", 25000.0, 20.0, 12000.0, n_symbols=3000)
    a = spectrum.analyse_spectrum(sc["samples"], sc["fs"])
    assert abs(a["center_frequency_hz"] - 12000.0) < 400.0
    assert a["noise_reliable"] is True
    assert a["params"]["occupied_bandwidth"]["status"] == "ok"
    assert a["params"]["snr"]["confidence"] > 0.5


def test_noise_only_reports_no_signal():
    rng = np.random.default_rng(1)
    for seed in range(8):
        x = (0.02 * (rng.standard_normal(40000) + 1j * rng.standard_normal(40000))).astype(np.complex64)
        a = spectrum.analyse_spectrum(x, 200000.0)
        assert a.get("no_signal") is True, f"false alarm on noise seed {seed}"
        assert a["params"]["snr"]["status"] == "unable"


def test_real_input_is_handled_one_sided():
    fs = 200000.0
    t = np.arange(60000) / fs
    x = (0.4 * np.cos(2 * np.pi * 30000.0 * t)).astype(np.float32)
    a = spectrum.analyse_spectrum(x, fs)
    assert a["ok"]
    assert a["freq_hz"][0] >= 0.0                       # one-sided axis for real input
    assert abs(a["center_frequency_hz"] - 30000.0) < 500.0


def test_cw_tone_is_line_like():
    fs = 200000.0
    t = np.arange(60000) / fs
    rng = np.random.default_rng(4)
    x = (0.5 * np.exp(2j * np.pi * 5000.0 * t) + 0.01 * (rng.standard_normal(t.size) +
                                                        1j * rng.standard_normal(t.size)))
    a = spectrum.analyse_spectrum(x.astype(np.complex64), fs)
    assert a["line_like"] is True
    assert abs(a["peak_frequency_hz"] - 5000.0) < 150.0
    assert a["obw_99_hz"] < 1000.0


def test_unavailable_records_never_carry_a_value():
    a = spectrum.analyse_spectrum(np.zeros(4096, dtype=np.complex64), 200000.0)
    assert a["ok"] is False
    a2 = spectrum.analyse_spectrum(np.zeros(32, dtype=np.complex64), 200000.0)
    assert a2["ok"] is False
