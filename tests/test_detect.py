"""Multi-signal detection: fidelity on known mixes and false-alarm behaviour on noise."""
import numpy as np
import pytest

from dsp import detect, scenarios


def _nearest(records, f_hz):
    return min(records, key=lambda r: abs(r["center_frequency_hz"] - f_hz))


def test_mix_a_finds_all_three_emissions():
    sc = scenarios.multi_signal_mix("A")
    res = detect.detect_signals(sc["samples"], sc["fs"])
    assert res["ok"]
    assert len(res["signals"]) == 3, [s["center_frequency_hz"] for s in res["signals"]]
    cw = _nearest(res["signals"], 60000.0)
    qpsk = _nearest(res["signals"], -30000.0)
    bpsk = _nearest(res["signals"], 25000.0)
    assert abs(cw["center_frequency_hz"] - 60000.0) < 500.0
    assert abs(qpsk["center_frequency_hz"] + 30000.0) < 1500.0
    assert abs(bpsk["center_frequency_hz"] - 25000.0) < 1500.0
    assert abs(qpsk["snr_db"] - 18.0) < 4.0
    assert abs(bpsk["snr_db"] - 12.0) < 4.0
    assert cw["bandwidth_hz"] < 4000.0
    for r in res["signals"]:
        assert 0.0 <= r["confidence"] <= 1.0
        assert r["evidence"]


def test_mix_b_ranks_by_strength():
    sc = scenarios.multi_signal_mix("B")
    res = detect.detect_signals(sc["samples"], sc["fs"])
    assert res["ok"] and len(res["signals"]) >= 2
    strongest = max(res["signals"], key=lambda r: r["snr_db"])
    assert abs(strongest["center_frequency_hz"]) < 2000.0
    ranked = detect.rank_signals(res["signals"])
    assert ranked[0]["rank_score"] >= ranked[-1]["rank_score"]


def test_gated_burst_detected_with_duty_cycle():
    sc = scenarios.gated_burst()
    res = detect.detect_signals(sc["samples"], sc["fs"])
    assert res["ok"]
    burst = _nearest(res["signals"], 20000.0)
    assert abs(burst["center_frequency_hz"] - 20000.0) < 1500.0
    assert burst["duty_cycle"] < 0.6, burst["duty_cycle"]
    assert burst["n_bursts"] >= 2


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_noise_only_has_no_false_alarms(seed):
    sc = scenarios.noise_only(seed=seed)
    res = detect.detect_signals(sc["samples"], sc["fs"])
    assert res["ok"]
    assert res["signals"] == [], [s["center_frequency_hz"] for s in res["signals"]]
    assert res["notes"]


def test_short_record_does_not_crash():
    res = detect.detect_signals(np.zeros(64, dtype=np.complex64), 200000.0)
    assert res["ok"] is False and "insufficient" in res["message"]


def test_cancellation_is_honoured():
    sc = scenarios.multi_signal_mix("A")
    res = detect.detect_signals(sc["samples"], sc["fs"], cancelled=lambda: True)
    assert res.get("cancelled") is True
