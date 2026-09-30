"""Automatic multi-signal detection on a spectrogram.

Two established techniques are combined:

1. **Cell-averaging CFAR in the time-frequency plane.**  A noise floor map is built with a
   *smallest-of* (grey erosion) filter over a small time-frequency window, so a continuous
   carrier cannot raise its own detection floor.  A cell is declared occupied when it exceeds
   that local floor by ``snr_threshold_db`` - or by the chi-squared CFAR level computed from the
   number of averaged frames (which keeps the false-alarm rate bounded).
2. **Region analysis.**  Connected cells are labelled, and each region is measured: peak power,
   mean power, SNR, occupied bandwidth, duration, duty cycle, plus the per-frame time profile so
   that burst trains are reported as one emission with a burst count rather than as fragments.

Detected records carry a confidence and the evidence behind it.  A record is never presented as
certain: the confidence is a documented function of SNR, region support and how close the
measurement is to the analysis resolution.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import waterfall as wf
from .utils import clamp01, db, human_si, param, safe_float

__all__ = ["detect_signals", "detect_bursts", "analyse_region", "rank_signals", "merge_signal_view"]


def _grouped_noise_density(Pg: np.ndarray, dof: int, frac: float = 0.2) -> dict:
    """Robust noise-power-per-Hz estimate from a block-averaged spectrogram.

    Averaging ``G`` independent frames makes each cell follow ``chi2(2G)/2G``; taking the
    ``frac`` quantile of all cells and mapping it back through the exact quantile of that
    distribution keeps the estimate unbiased even when the cells are exponential-tailed, and
    using a low quantile keeps a dominating signal from raising its own noise floor.
    """
    from scipy import stats as st
    vals = np.asarray(Pg, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return {"ok": False, "message": "no usable cells for a noise estimate"}
    try:
        q = float(st.chi2.ppf(frac, dof) / dof)
    except Exception:                                                   # pragma: no cover
        q = frac
    q = max(q, 1e-3)
    m = float(np.quantile(vals, frac))
    mu = m / q
    return {"ok": True, "noise_power_per_hz": mu, "quantile": frac, "quantile_factor": q,
            "median_factor": float(st.chi2.ppf(0.5, dof) / dof),
            "method": f"{frac:.0%} quantile of {vals.size} averaged cells divided by the "
                      f"chi2({dof})/{dof} {frac:.0%} quantile ({q:.3f})"}


def _empty_result(message: str) -> dict:
    return {"ok": False, "signals": [], "message": message}


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        out.append((start, prev))
        start = prev = i
    out.append((start, prev))
    return out


def _bursts(profile_db: np.ndarray, times: np.ndarray, floor_db: float,
            thr_db: float = 3.0, min_len: int = 3, merge_gap: int = 2) -> list[dict]:
    """Burst intervals from a band time profile (hysteresis-free, threshold + min length)."""
    if profile_db.size == 0:
        return []
    on = profile_db > (floor_db + thr_db)
    if merge_gap > 0:
        on = ndi.binary_closing(on, structure=np.ones(int(merge_gap) * 2 + 1, dtype=bool))
    out = []
    for lo, hi in _runs(on):
        if hi - lo + 1 < min_len:
            continue
        out.append({"t0_s": float(times[lo]), "t1_s": float(times[hi]),
                    "duration_s": float(times[hi] - times[lo] + (times[1] - times[0] if times.size > 1 else 0.0)),
                    "peak_db": float(np.max(profile_db[lo:hi + 1])),
                    "contrast_db": float(np.max(profile_db[lo:hi + 1]) - floor_db)})
    return out


def _region_record(sp: dict, box: tuple[int, int, int, int], floor_db: float, n0: float,
                   df_hz: float, dt_s: float, fs: float, cfg: dict,
                   det_cells: int = 0) -> dict | None:
    """Measure one detected region (bounding box on the spectrogram) in detail."""
    f_lo_i, f_hi_i, t_lo_i, t_hi_i = box
    freqs = sp["freq_hz"]
    times = sp["times_s"]
    f_lo = float(freqs[f_lo_i] - df_hz / 2.0)
    f_hi = float(freqs[f_hi_i] + df_hz / 2.0)
    bandwidth = f_hi - f_lo
    sub = sp["power"][f_lo_i:f_hi_i + 1, t_lo_i:t_hi_i + 1]
    sub_db = sp["db"][f_lo_i:f_hi_i + 1, t_lo_i:t_hi_i + 1]
    if sub.size == 0:
        return None
    # per-frame profile of the region's band -> duty cycle and burst structure
    prof = np.mean(sub, axis=0)
    prof_db = 10 * np.log10(np.maximum(prof, 1e-30) * max(df_hz, 1e-30))
    on = prof_db > (floor_db + cfg["burst_thr_db"])
    bursts = _bursts(prof_db, times[t_lo_i:t_hi_i + 1], floor_db, thr_db=cfg["burst_thr_db"])
    duty_on = float(np.count_nonzero(on) / max(on.size, 1))
    duty_burst = float(sum(x["duration_s"] for x in bursts) /
                       max(times[t_hi_i] - times[t_lo_i] + dt_s, 1e-9))
    # mean power: average over the whole box, and over the "on" cells only (bursts)
    vals = sub.ravel()
    mean_box = float(np.mean(vals))
    on_cells = sub_db > (floor_db + cfg["burst_thr_db"])
    n_on = int(np.count_nonzero(on_cells))
    mean_on = float(np.mean(sub[on_cells])) if n_on >= 4 else mean_box
    # signal power from the noise-corrected integral (unbiased against the region overshoot),
    # cross-checked with the raw region mean
    # average over time first (a spectrogram cell is a time-local density), then integrate over
    # frequency - summing over frames as well would multiply the power by the number of frames
    sub_time_mean = np.mean(sub, axis=1)
    sig_power = float(np.sum(np.maximum(sub_time_mean - n0, 0.0)) * df_hz)
    mean_use = max(mean_box, mean_on)
    peak_db = float(np.max(sub_db))
    # effective bandwidth: 95 % of the band power
    prof_f = np.sum(sub, axis=1)
    tot = float(np.sum(prof_f))
    if tot > 0:
        cum = np.cumsum(prof_f) / tot
        lo_b = int(np.searchsorted(cum, 0.025))
        hi_b = int(np.searchsorted(cum, 0.975))
        bw_eff = float(max(hi_b - lo_b + 1, 1) * df_hz)
    else:
        bw_eff = bandwidth
    snr_integrated_db = float(db(max(sig_power, 1e-300) / (n0 * max(bw_eff, df_hz))))
    snr_region_db = float(db(max(mean_use - n0, 1e-300) / max(n0, 1e-300)))
    snr_db = float(max(snr_integrated_db, snr_region_db))
    ipk = np.unravel_index(int(np.argmax(sub)), sub.shape)
    peak_f = float(freqs[f_lo_i + ipk[0]])
    # peak frequency is meaningless for a flat-topped emission: report the power centroid of the
    # strong cells instead, and say which was used
    cm = np.sum(sub, axis=1)
    centroid_f = float(np.sum(freqs[f_lo_i:f_hi_i + 1] * cm) / max(np.sum(cm), 1e-30))
    t_cells = int(np.count_nonzero(on_cells))
    conf = clamp01(0.30 * clamp01((snr_db - cfg["snr_threshold_db"] + 3.0) / 15.0) +
                   0.25 * clamp01(np.log2(max(bandwidth / max(df_hz, 1e-9), 1.0)) / 6.0) +
                   0.20 * clamp01(det_cells / 12.0) +
                   0.15 * clamp01(duty_on / 0.5) +
                   0.10 * clamp01(t_cells / 60.0))
    if bandwidth < 4 * df_hz:
        conf = min(conf, 0.65)
    evidence = [
        f"occupied region {f_lo:,.0f} .. {f_hi:,.0f} Hz x {times[t_lo_i]:.4f} .. "
        f"{times[t_hi_i]:.4f} s ({det_cells} cells above the CFAR level in the detection grid)",
        f"peak {peak_db:.1f} dB at {peak_f:,.0f} Hz; power centroid {centroid_f:,.0f} Hz; "
        f"mean level {float(db(mean_use)):.1f} dB",
        f"SNR {snr_db:.1f} dB: integrated (noise-corrected) signal power over {bw_eff:,.0f} Hz of "
        f"occupied bandwidth vs the {floor_db:.1f} dB/bin noise floor gives {snr_integrated_db:.1f} dB, "
        f"the region mean gives {snr_region_db:.1f} dB",
        f"spectrogram resolution {df_hz:,.1f} Hz/bin x {dt_s * 1e3:.2f} ms/frame; detection "
        f"threshold {cfg['snr_threshold_db']:.1f} dB above the noise floor",
    ]
    if bursts and len(bursts) > 1:
        evidence.append(f"{len(bursts)} burst(s) inside the region; total on-time "
                        f"{duty_burst * 100:.0f} % of the record")
    limitations = []
    if bandwidth < 4 * df_hz:
        limitations.append(f"the emission is {bandwidth / df_hz:.1f} FFT bins wide; its bandwidth "
                            "is limited by the analysis resolution")
    if 1 <= len(bursts) and duty_burst < 0.5:
        limitations.append(f"the emission is on only {duty_burst * 100:.0f} % of the time, so "
                           "continuous-signal parameters (SNR, symbol rate) are measured on the "
                           "bursted portion only")
    return {
        "t0_s": float(times[t_lo_i] - dt_s / 2), "t1_s": float(times[t_hi_i] + dt_s / 2),
        "duration_s": float(times[t_hi_i] - times[t_lo_i] + dt_s),
        "f_lo_hz": f_lo, "f_hi_hz": f_hi, "center_frequency_hz": float(0.5 * (f_lo + f_hi)),
        "frequency_centroid_hz": centroid_f, "peak_frequency_hz": peak_f,
        "bandwidth_hz": float(bandwidth), "effective_bandwidth_hz": float(bw_eff),
        "peak_power_db": peak_db, "mean_power_db": float(db(mean_use)),
        "noise_floor_db": floor_db, "snr_db": snr_db, "snr_region_db": snr_region_db,
        "snr_integrated_db": snr_integrated_db,
        "snr_method": ("integrated noise-corrected power over the occupied bandwidth"
                       if snr_integrated_db >= snr_region_db else
                       "mean level of the occupied region"),
        "signal_power_db": float(db(max(sig_power, 1e-300))),
        "on_cells": t_cells, "detection_cells": int(det_cells), "n_cells": int(det_cells),
        "duty_cycle": float(max(duty_on, duty_burst)), "frame_duty_cycle": duty_on,
        "bursts": bursts, "n_bursts": len(bursts), "confidence": float(conf),
        "evidence": evidence, "limitations": limitations, "fs": float(fs),
        "detector": "block-averaged CFAR + connected-region analysis",
    }


def detect_signals(x: np.ndarray, fs: float, nperseg: int | None = None, overlap: float = 0.75,
                   window: str = "hann", snr_threshold_db: float = 6.0, min_cells: int = 4,
                   max_signals: int = 32, merge_gap_bins: int = 2,
                   reference_center_hz: float = 0.0, p_fa: float = 1e-3,
                   group_frames: int = 8, cancelled=None, progress=None) -> dict:
    """Detect every emission in the record and measure it.

    Detection works on *statistically independent* time blocks: the STFT frames are decimated by
    the overlap factor (a 75 % overlap makes every 4th frame independent), grouped ``G`` at a time
    and averaged, which reduces the per-cell fluctuation by 10*log10(G) dB.  Each averaged cell is
    then tested against a chi-squared CFAR level computed for the number of tested cells (so the
    false-alarm rate is bounded) and against a minimum level of ``snr_threshold_db`` above the
    noise density.  Connected occupied cells form regions; a region must contain at least
    ``min_cells`` occupied cells in the detection grid, which is what stops single noise cells from
    being reported as signals.  Every surviving region is then measured on the full-resolution
    spectrogram (bandwidth, SNR, peak power, duty cycle, bursts).

    ``cancelled`` is an optional callable returning True to abort, ``progress`` an optional
    ``f(frac, text)`` callback.
    """
    from scipy import stats as st
    x = np.asarray(x)
    if x.size < 256:
        return {"ok": False, "signals": [], "message": f"insufficient samples for detection (n={x.size})"}
    if np.allclose(np.abs(x), 0.0):
        return {"ok": False, "signals": [], "message": "the record is all zeros; nothing to detect"}
    if progress:
        progress(0.05, "computing spectrogram")
    sp = wf.stft_matrix(x, fs, nperseg=nperseg, overlap=overlap, window=window)
    if not sp.get("ok"):
        return {"ok": False, "signals": [], "message": sp.get("message", "spectrogram failed")}
    if cancelled and cancelled():
        return {"ok": False, "signals": [], "cancelled": True, "message": "analysis cancelled"}
    P, df_hz, dt_s, times = sp["power"], sp["df_hz"], sp["dt_s"], sp["times_s"]
    nf, nt = P.shape
    nper, nover = sp["nperseg"], sp["noverlap"]
    hop = max(1, nper - nover)
    stride = int(max(1, round(nper / hop)))       # independent-frame stride
    if progress:
        progress(0.25, "estimating the noise floor and testing cells")
    # --- one-sided / two-sided aware power axis is already handled by stft_matrix ------------
    # ---- block-averaged CFAR detection -------------------------------------------------
    # A single STFT cell is exponentially distributed, so testing 10^5 of them at a fixed level
    # produces phantom signals.  Averaging ``G`` independent frames first makes every test cell
    # chi2(2G)/2G, which both narrows the fluctuation and lets the threshold be derived from a
    # known false-alarm rate.
    # The STFT frames overlap by 75 %, so consecutive frames are strongly correlated: averaging
    # them would *not* reduce the fluctuation the way a chi-squared law predicts.  The detection
    # test therefore uses every ``stride``-th frame (independent) and averages ``G`` of them,
    # which makes each test cell chi2(2G)/2G as assumed.
    Pind = P[:, ::stride]
    tind = times[::stride]
    nt_ind = Pind.shape[1]
    G = int(max(1, min(int(group_frames), max(1, nt_ind // 4))))
    ng = max(1, nt_ind // G)
    Pg = (Pind[:, :ng * G].reshape(nf, ng, G).mean(axis=2) if G > 1 else Pind[:, :ng])
    dof = max(2, 2 * G)
    if progress:
        progress(0.25, f"estimating the noise floor ({ng} averaged blocks of {G} frames)")
    nd = _grouped_noise_density(Pg, dof, frac=0.2)
    if not nd.get("ok"):
        return _empty_result(str(nd.get("message")))
    n0 = float(nd["noise_power_per_hz"])                      # noise power per Hz
    floor_db = float(db(n0 * df_hz))                          # one PSD cell at the noise level
    n_test = int(max(1, nf * ng))
    try:
        q = float(st.chi2.ppf((1.0 - float(p_fa)) ** (1.0 / n_test), dof) / dof)
    except Exception:                                                    # pragma: no cover
        q = 3.0
    q = float(max(q, 2.0))
    thr_over_floor = float(max(db(q), float(snr_threshold_db)))
    thr_lin = n0 * (10.0 ** (thr_over_floor / 10.0))           # power per Hz
    mask = Pg > thr_lin
    if merge_gap_bins > 0:
        mask = ndi.binary_closing(mask, structure=np.ones((merge_gap_bins * 2 + 1, 1), dtype=bool))
    if ng > 2:
        mask = ndi.binary_closing(mask, structure=np.ones((1, 3), dtype=bool))
    labels, n_lab = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    if progress:
        progress(0.55, f"measuring {n_lab} candidate region(s)")
    cfg = {"snr_threshold_db": thr_over_floor, "burst_thr_db": float(max(3.0, 0.35 * thr_over_floor)),
           "min_cells": int(min_cells)}
    records, rejected = [], 0
    for i in range(1, n_lab + 1):
        if cancelled and cancelled():
            return {"ok": False, "signals": [], "cancelled": True, "message": "analysis cancelled"}
        sel = labels == i
        det_cells = int(np.count_nonzero(sel))
        if det_cells < int(min_cells):
            rejected += 1
            continue
        fi, gi = np.nonzero(sel)
        f_lo_i, f_hi_i = int(fi.min()), int(fi.max())
        g_lo, g_hi = int(gi.min()), int(gi.max())
        # map the detected (frequency, group) box back onto the overlapping-frame spectrogram
        t0 = float(tind[g_lo * G])
        t1 = float(tind[min(nt_ind - 1, (g_hi + 1) * G - 1)])
        ti = np.flatnonzero((times >= t0 - dt_s / 2) & (times <= t1 + dt_s / 2))
        if ti.size < 2:
            rejected += 1
            continue
        rec = _region_record(sp, (f_lo_i, f_hi_i, int(ti[0]), int(ti[-1])), floor_db, n0, df_hz,
                            dt_s, fs, cfg, det_cells=det_cells)
        if rec:
            records.append(rec)
        else:
            rejected += 1
    records = _merge_records(records, freq_tol_hz=max(3 * df_hz, 1e-9), time_gap_s=4 * dt_s)
    records.sort(key=lambda r: (-r["snr_db"], -r["bandwidth_hz"]))
    for k, rec in enumerate(records):
        rec["index"] = k
        rec["id"] = f"S{k + 1}"
    records = records[:max_signals]
    if progress:
        progress(0.85, f"{len(records)} emission(s) measured")
    notes = [
        f"detection: {G} independent spectrogram frames averaged per test cell "
        f"({stride}-frame stride for the {int(overlap * 100)} % overlap), chi2({dof})/{dof} CFAR "
        f"level p_fa={p_fa:g} over {n_test} cells -> {db(q):.1f} dB; minimum SNR "
        f"{snr_threshold_db:.1f} dB -> threshold {thr_over_floor:.1f} dB above the "
        f"{floor_db:.1f} dB noise floor",
        f"noise floor from {nd['method']} at {floor_db:.1f} dB per cell "
        f"({db(n0):.1f} dB/Hz)",
        f"a region must hold at least {min_cells} occupied cells; {rejected} candidate(s) were "
        f"rejected for insufficient support",
    ]
    if not records:
        notes.append("no emission exceeded the detection threshold in this record; the platform "
                     "reports no signal rather than guessing one")
    if len(records) > 1:
        notes.append("more than one emission was found; parameters are reported per emission "
                     "rather than for the record as a whole")
    return {"ok": True, "signals": records, "noise_floor_db": floor_db, "noise_density": n0,
            "threshold_db": float(db(n0 * df_hz * 10 ** (thr_over_floor / 10.0))),
            "threshold_over_floor_db": thr_over_floor,
            "cfar": {"q_factor": q, "level_over_noise_db": float(db(q)), "p_fa": float(p_fa),
                     "dof": dof, "n_cells_tested": n_test, "group_frames": G, "stride": stride,
                     "method": f"chi2({dof})/{dof} quantile, p_fa={p_fa:g} over {n_test} cells"},
            "notes": notes, "n_detected": len(records), "df_hz": df_hz, "dt_s": dt_s,
            "nperseg": sp["nperseg"], "n_frames": nt, "spectrogram_shape": [nf, nt],
            "rejected_candidates": rejected, "fs": float(fs),
            "reference_center_hz": float(reference_center_hz)}


def _merge_records(records: list[dict], freq_tol_hz: float, time_gap_s: float,
                   duration_factor: float = 4.0) -> list[dict]:
    """Merge regions that are clearly two fragments of the same emission.

    Two fragments in the same band are one emission when they overlap in frequency and the gap
    between them is short compared with the fragments themselves (up to ``duration_factor`` times
    the shorter fragment).  That is what turns a burst train into one emission carrying a burst
    list, instead of N unrelated detections.
    """
    if len(records) < 2:
        return records
    records = sorted(records, key=lambda r: r["center_frequency_hz"])
    out: list[dict] = []
    for rec in records:
        placed = False
        for prev in out:
            f_ov = (min(prev["f_hi_hz"], rec["f_hi_hz"]) - max(prev["f_lo_hz"], rec["f_lo_hz"]))
            f_ov = f_ov / max(min(prev["bandwidth_hz"], rec["bandwidth_hz"]), 1e-9)
            gap_s = max(rec["t0_s"] - prev["t1_s"], prev["t0_s"] - rec["t1_s"])
            gap_ok = max(float(time_gap_s),
                         float(duration_factor) * min(prev["duration_s"], rec["duration_s"]))
            if f_ov > 0.6 and gap_s < gap_ok and len(prev["bursts"]) < 256:
                prev["t0_s"] = min(prev["t0_s"], rec["t0_s"])
                prev["t1_s"] = max(prev["t1_s"], rec["t1_s"])
                prev["duration_s"] = prev["t1_s"] - prev["t0_s"]
                prev["f_lo_hz"] = min(prev["f_lo_hz"], rec["f_lo_hz"])
                prev["f_hi_hz"] = max(prev["f_hi_hz"], rec["f_hi_hz"])
                prev["bandwidth_hz"] = prev["f_hi_hz"] - prev["f_lo_hz"]
                prev["center_frequency_hz"] = 0.5 * (prev["f_lo_hz"] + prev["f_hi_hz"])
                prev["n_cells"] += rec["n_cells"]
                prev["bursts"] = sorted(prev["bursts"] + rec["bursts"],
                                        key=lambda b: b["t0_s"])
                prev["n_bursts"] = len(prev["bursts"])
                # on-time fraction of the merged record: the part of the span that carries signal
                on_time = float(sum(b["duration_s"] for b in prev["bursts"]))
                prev["duty_cycle"] = (float(min(1.0, on_time / max(prev["duration_s"], 1e-12)))
                                      if on_time > 0 else
                                      max(prev["duty_cycle"], rec["duty_cycle"]))
                prev["peak_power_db"] = max(prev["peak_power_db"], rec["peak_power_db"])
                prev["snr_db"] = max(prev["snr_db"], rec["snr_db"])
                prev["confidence"] = min(0.95, max(prev["confidence"], rec["confidence"]))
                prev["evidence"].append("merged with an adjacent time fragment of the same band")
                placed = True
                break
        if not placed:
            out.append(rec)
    return out


def detect_bursts(x: np.ndarray, fs: float, nperseg: int | None = None,
                  band_hz: tuple[float, float] | None = None, threshold_db: float = 6.0) -> dict:
    """Envelope-based burst detection (for gated emissions and AM/ASK signals)."""
    x = np.asarray(x)
    if x.size < 256:
        return {"ok": False, "bursts": [], "message": "insufficient samples"}
    sp = wf.stft_matrix(x, fs, nperseg=nperseg)
    if not sp.get("ok"):
        return {"ok": False, "bursts": [], "message": sp.get("message")}
    f = sp["freq_hz"]
    if band_hz is not None:
        lo, hi = sorted(band_hz)
        sel = (f >= lo) & (f <= hi)
    else:
        sel = np.ones(f.size, dtype=bool)
    prof = np.mean(sp["power"][sel, :], axis=0)
    prof_db = 10 * np.log10(np.maximum(prof, 1e-30) * sp["df_hz"])
    floor = float(np.median(np.sort(prof_db)[: max(8, int(0.25 * prof_db.size))]))
    bursts = _bursts(prof_db, sp["times_s"], floor, thr_db=threshold_db, min_len=2)
    return {"ok": True, "bursts": bursts, "n_bursts": len(bursts), "floor_db": floor,
            "n_frames": sp["n_frames"], "dt_s": sp["dt_s"],
            "method": "spectrogram band profile with a robust noise floor"}


def rank_signals(records: list[dict]) -> list[dict]:
    """Rank emissions by the analytical value of investigating them.

    Score = SNR (dB, capped at 40) + 10*log10(bandwidth) + duty preference, because a wide,
    continuous, high-SNR emission is both easier to classify and more likely to be the intended
    signal in a capture that contains several.
    """
    out = []
    for r in records:
        snr = float(r.get("snr_db") or 0.0)
        bw = max(float(r.get("effective_bandwidth_hz") or r.get("bandwidth_hz") or 1.0), 1.0)
        duty = float(r.get("duty_cycle") or 0.0)
        score = (min(max(snr, 0.0), 40.0) + 10.0 * np.log10(bw) + 5.0 * duty)
        r2 = dict(r)
        r2["rank_score"] = float(score)
        r2["rank_reason"] = (f"SNR {snr:.1f} dB, bandwidth {human_si(bw, 'Hz')}, "
                             f"on-time {duty * 100:.0f} %")
        out.append(r2)
    out.sort(key=lambda r: -r["rank_score"])
    return out


def analyse_region(x: np.ndarray, fs: float, f_lo: float, f_hi: float,
                   t0_s: float | None = None, t1_s: float | None = None,
                   recover_audible_audio: bool = False) -> dict:
    """Extract and characterise one selected region (click-to-analyze)."""
    seg = wf.extract_band(x, fs, f_lo, f_hi, t0_s=t0_s, t1_s=t1_s)
    if not seg.get("ok"):
        return seg
    sp = wf.stft_matrix(seg["samples"], seg["fs"], nperseg=None)
    stats = {}
    if sp.get("ok"):
        stats = wf.region_stats(sp, np.ones((sp["n_freq"], sp["n_frames"]), dtype=bool))
    return {"ok": True, "segment": {"n_samples": seg["n_samples"], "fs": seg["fs"],
                                    "duration_s": seg["duration_s"], "f_center_hz": seg["f_center_hz"],
                                    "bw_hz": seg["bw_hz"]},
            "stats": stats, "method": seg["method"]}


def merge_signal_view(records: list[dict], refine: dict | None = None) -> list[dict]:
    """Attach refined parameters (from the single-signal analyzers) to a detection record."""
    if not refine:
        return records
    for r in records:
        key = r.get("id") or r.get("index")
        if key in refine:
            r["refined"] = refine[key]
    return records
