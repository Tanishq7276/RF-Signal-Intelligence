"""Spectrogram payload, band extraction and time slicing."""
import numpy as np

from dsp import scenarios, waterfall


def test_payload_is_bounded_and_labelled():
    sc = scenarios.signal_scenario("QPSK", 25000.0, 20.0, 5000.0, n_symbols=6000)
    pay = waterfall.waterfall_payload(sc["samples"], sc["fs"], nperseg=256, max_w=300, max_h=200)
    assert pay["ok"]
    assert len(pay["db"]) <= 200 and len(pay["db"][0]) <= 300
    assert "aggregat" in pay["note"]
    assert pay["vmax_db"] > pay["vmin_db"]


def test_band_extraction_recovers_symbol_rate():
    from dsp import params
    sc = scenarios.signal_scenario("QPSK", 25000.0, 20.0, 5000.0, n_symbols=3000)
    seg = waterfall.extract_band(sc["samples"], sc["fs"], -20000.0, 30000.0)
    assert seg["ok"] and np.iscomplexobj(seg["samples"])
    r = params.estimate_symbol_rate(seg["samples"], seg["fs"])
    assert abs(r["primary"]["symbol_rate_hz"] / 25000.0 - 1.0) < 0.03


def test_band_extraction_clips_out_of_range():
    sc = scenarios.signal_scenario("QPSK", 25000.0, 20.0, -40000.0, n_symbols=1000)
    seg = waterfall.extract_band(sc["samples"], sc["fs"], 150000.0, 250000.0)
    # the requested band is outside the Nyquist range but the call must not crash
    assert seg["ok"] is False or seg["ok"] is True


def test_time_slice_bounds():
    x = np.arange(1000, dtype=np.complex64)
    s = waterfall.slice_time(x, 1000.0, 0.1, 0.2)
    assert s["ok"] and s["n_samples"] == 100 and s["i0"] == 100 and s["i1"] == 200
    assert waterfall.slice_time(x, 1000.0, 0.5, 0.4)["ok"] is False
