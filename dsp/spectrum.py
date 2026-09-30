"""Spectrum / PSD analysis: occupied bandwidth, noise floor, SNR, carrier and peak frequency.

Every reported number carries a status, a confidence, the method that produced it and the
evidence behind it (see :func:`dsp.utils.param`).  Nothing is invented: when the estimate is
not supportable by the data the record is returned with ``status="unable"`` and the reason is
put in ``limitations`` so the UI can display ``Unable to estimate reliably``.

The estimator conventions used here are the usual spectrum-analysis ones:

* **PSD** - Welch averaged periodogram, ``scaling="density"``, so the values are power per Hz
  (two-sided for complex input, one-sided for real input).
* **Noise floor** - median of the PSD bins (median spectral estimator).  It is robust to the
  signal occupying a minority of the observed band; when the signal occupies most of the band
  the estimate is flagged as unreliable instead of being silently reported.
* **Occupied bandwidth** - the narrowest band around the peak that contains ``frac`` of the
  noise-corrected power, with interpolation on the cumulative power curve.
* **SNR** - signal power (integrated noise-corrected PSD over the occupied band) divided by the
  noise power contained in the same band (``noise_density * bandwidth``).
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sigproc
from scipy import stats as st

from .utils import (db, downsample_complex, estimate_noise_floor_db, human_si, param,
                    parabolic_peak, safe_float, clamp01, unavailable, welch_style_smooth)
from .preprocess import make_window

DEFAULT_WINDOWS = ["hann", "hamming", "blackman", "flattop", "rectangular"]

__all__ = [
    "welch_psd", "psd_to_db", "estimate_noise_density", "analyse_spectrum",
    "band_power_db", "envelope_spectrum", "estimate_am_index", "instantaneous_frequency",
    "band_snr", "region_spectrum_params",
]


# ---------------------------------------------------------------------------------------
# basic estimators
# ---------------------------------------------------------------------------------------
def _auto_nperseg(n: int, nperseg: int | None = None) -> int:
    if nperseg:
        nper = int(nperseg)
    else:
        nper = int(min(4096, max(256, n // 8)))
    nper = int(min(max(64, nper), n))
    # round down to a power of two for predictable behaviour, but never below 64
    if nper >= 64:
        nper = 1 << int(np.floor(np.log2(nper)))
    return int(max(64, min(nper, n)))


def welch_psd(x: np.ndarray, fs: float, nperseg: int | None = None,
              window: str = "hann", overlap: float = 0.5) -> dict:
    """Welch PSD (power per Hz).  Frequency axis is fftshifted for complex input."""
    x = np.asarray(x)
    n = x.size
    if n < 32:
        return {"ok": False, "message": f"insufficient samples for a PSD estimate (n={n})"}
    nper = _auto_nperseg(n, nperseg)
    nover = int(max(0, min(nper - 1, round(nper * float(np.clip(overlap, 0.0, 0.9))))))
    win = make_window(window, nper)
    real_input = not np.iscomplexobj(x)
    try:
        f, P = sigproc.welch(x.astype(np.complex128 if not real_input else np.float64),
                             fs=fs, window=win, nperseg=nper, noverlap=nover,
                             detrend="constant", return_onesided=real_input,
                             scaling="density")
    except Exception as exc:                                        # pragma: no cover
        return {"ok": False, "message": f"PSD estimate failed: {exc}"}
    if not np.iscomplexobj(x):
        f = f.copy()
    else:
        f = np.fft.fftshift(f)
        P = np.fft.fftshift(P)
    return {
        "ok": True, "freq_hz": f, "psd": P, "df_hz": float(fs / nper),
        "nperseg": nper, "noverlap": nover, "n_segments": int(max(1, (n - nover) // (nper - nover))),
        "window": window, "onesided": bool(real_input), "fs": float(fs),
        "resolution_hz": float(fs / nper), "estimator": "welch-periodogram (density)",
    }


def psd_to_db(P: np.ndarray, df_hz: float = 1.0) -> np.ndarray:
    """PSD (power/Hz) to dB relative to 1.0 (``df`` folded in so bin integrals match)."""
    return 10.0 * np.log10(np.maximum(np.asarray(P, dtype=np.float64), 1e-30) * max(df_hz, 1e-30))


def _smooth_psd(P: np.ndarray, n: int = 5) -> np.ndarray:
    if n <= 1 or P.size < n:
        return P
    return welch_style_smooth(P, n)


def estimate_noise_density(f: np.ndarray, P: np.ndarray, signal_mask: np.ndarray | None = None,
                           df_hz: float | None = None, n_segments: int | None = None) -> dict:
    """Robust noise power-density estimate from a Welch PSD (median spectral estimator).

    The quietest 25 % of the bins are used, and the measured quantile is corrected for the
    chi-squared (2K degrees of freedom for K averaged segments) statistics of a periodogram bin -
    without that correction the floor would be biased low by several dB and every SNR would come
    out too high.

    Both conventions are returned explicitly, because mixing them shifts every threshold:

    * ``noise_density`` / ``noise_density_db_hz`` - power per Hz (used for SNR integrals)
    * ``noise_floor_db``                          - level of one PSD bin in dBFS (thresholds/display)
    """
    P = np.asarray(P, dtype=np.float64)
    if P.size < 8:
        return {"ok": False, "message": "not enough PSD bins for a noise estimate"}
    if df_hz is None:
        df_hz = float(f[1] - f[0]) if f.size > 1 else 1.0
    K = int(n_segments) if n_segments else 1
    dof = max(2, 2 * K)
    # quantile of the ordered statistic we are about to measure
    p_order = 0.125
    try:
        q_order = float(st.chi2.ppf(p_order, dof) / dof)
    except Exception:                                                   # pragma: no cover
        q_order = float(p_order)
    q_order = max(q_order, 1e-3)
    order = np.argsort(P)
    k = max(8, int(0.25 * P.size))
    quiet = P[order[:k]]
    meas = float(np.median(quiet)) if quiet.size else float(np.median(P))
    n0 = max(meas / q_order, 1e-300)
    med = float(np.median(P))
    frac_signal = 0.0
    if signal_mask is not None and signal_mask.size == P.size:
        frac_signal = float(np.mean(signal_mask))
    ratio_db = 10.0 * np.log10(max(n0, 1e-30) / max(med, 1e-30))
    reliable = bool(frac_signal < 0.5 and ratio_db > -12.0)
    return {"ok": True, "noise_density": n0,
            "noise_density_db_hz": float(db(n0)),
            "noise_floor_db": float(db(n0 * max(df_hz, 1e-30))),
            "df_hz": float(df_hz), "signal_bin_fraction": frac_signal,
            "reliable": reliable, "p25_median_linear": meas, "median_density": med,
            "quantile_correction": float(q_order), "n_segments": K, "dof": dof,
            "method": f"median of the quietest 25 % of PSD bins, corrected for the "
                      f"chi2({dof})/{dof} quantile at p={p_order:g}",
            "limitation": None if reliable else
            ("the signal occupies too much of the observed band for a reliable "
             "noise-floor estimate; provide a wider capture or a quiet reference band")}


def _cumulative_bandwidth(f: np.ndarray, P: np.ndarray, frac: float) -> dict:
    """Narrowest band around the peak holding ``frac`` of the total power."""
    if P.size < 4 or P.sum() <= 0:
        return {"ok": False}
    ipk = int(np.argmax(P))
    total = float(P.sum())
    # grow outwards from the peak, adding the larger neighbour first
    lo = hi = ipk
    acc = float(P[ipk])
    while acc < frac * total and (lo > 0 or hi < P.size - 1):
        left = P[lo - 1] if lo > 0 else -1.0
        right = P[hi + 1] if hi < P.size - 1 else -1.0
        if right >= left:
            hi += 1
            acc += float(P[hi])
        else:
            lo -= 1
            acc += float(P[lo])
    # linear interpolation on the edge bins for sub-bin resolution
    df = float(f[1] - f[0]) if f.size > 1 else 0.0
    frac_lo = f[lo] - df / 2.0
    frac_hi = f[hi] + df / 2.0
    return {"ok": True, "lo_hz": float(frac_lo), "hi_hz": float(frac_hi),
            "bandwidth_hz": float(frac_hi - frac_lo), "power_fraction": float(acc / total),
            "peak_bin": ipk, "peak_freq_hz": float(f[ipk])}


def _edge_bandwidth(f: np.ndarray, Pdb: np.ndarray, rel_db: float) -> dict:
    """Bandwidth relative to the peak (``rel_db`` below the peak level)."""
    if Pdb.size < 4:
        return {"ok": False}
    ipk = int(np.argmax(Pdb))
    thr = Pdb[ipk] + rel_db
    lo = ipk
    while lo > 0 and Pdb[lo - 1] >= thr:
        lo -= 1
    hi = ipk
    while hi < Pdb.size - 1 and Pdb[hi + 1] >= thr:
        hi += 1
    df = float(f[1] - f[0]) if f.size > 1 else 0.0
    return {"ok": True, "lo_hz": float(f[lo] - df / 2), "hi_hz": float(f[hi] + df / 2),
            "bandwidth_hz": float((hi - lo + 1) * df)}


def cfar_threshold_db(P: np.ndarray, n0: float, df_hz: float, n_segments: int,
                      p_fa: float = 1e-3) -> dict:
    """Cell-averaging false-alarm threshold for a Welch PSD.

    A Welch bin follows ``chi2(2K)/2K`` (K averaged segments).  Solving for the level whose
    probability of being exceeded by *any* of the N analysed bins equals ``p_fa`` gives a
    detection threshold with a known false-alarm rate - which is what stops noise-only
    records from producing phantom signals.
    """
    P = np.asarray(P, dtype=np.float64)
    nb = P.size
    K = max(1, int(n_segments))
    T = 2.0 * K
    try:
        q = float(st.chi2.ppf((1.0 - float(p_fa)) ** (1.0 / max(nb, 1)), T) / T)
    except Exception:
        q = 3.0
    q = max(q, 2.0)
    level = float(n0 * df_hz * q)
    return {"ok": True, "q_factor": float(q), "level_linear": level,
            "level_db": float(db(level)), "p_fa": float(p_fa), "n_bins": int(nb),
            "dof": int(T), "n_segments": K,
            "method": f"CFAR: chi2({T})/{T} quantile for p_fa={p_fa:g} over {nb} bins "
                      f"({K} averaged Welch segments)"}


def _find_regions(f: np.ndarray, Pdb: np.ndarray, floor_db: float, thr_db: float,
                  min_bins: int = 1, merge_bins: int = 2) -> list[dict]:
    """Connected regions of the PSD above the floor + threshold."""
    sm = _smooth_psd(Pdb, 5)
    mask = sm > (floor_db + thr_db)
    if not np.any(mask):
        return []
    idx = np.flatnonzero(mask)
    groups, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev <= merge_bins:
            prev = i
            continue
        groups.append((start, prev))
        start = prev = i
    groups.append((start, prev))
    out = []
    df = float(f[1] - f[0]) if f.size > 1 else 0.0
    for lo, hi in groups:
        if hi - lo + 1 < min_bins:
            continue
        seg = Pdb[lo:hi + 1]
        out.append({"bin_lo": int(lo), "bin_hi": int(hi), "lo_hz": float(f[lo] - df / 2),
                    "hi_hz": float(f[hi] + df / 2), "bandwidth_hz": float((hi - lo + 1) * df),
                    "peak_db": float(np.max(seg)),
                    "peak_freq_hz": float(f[lo + int(np.argmax(seg))]),
                    "snr_over_floor_db": float(np.max(seg) - floor_db)})
    out.sort(key=lambda r: r["peak_db"], reverse=True)
    return out


def cluster_regions(regions: list[dict], df: float, max_gap_bins: int = 8,
                    min_bins: int = 2) -> list[dict]:
    """Group per-bin CFAR regions into *emissions*.

    A single modulated carrier is often several spectral regions: an AM carrier with its
    sidebands, an FSK signal with two tones, an FM signal with a Bessel line comb, or a carrier
    with clock spurs.  Regions separated by less than ``max_gap_bins`` are therefore treated as
    one emission.  The caller sets the gap from the analysis resolution so the rule reads as
    "features closer than N bins belong to the same emission".
    """
    if not regions:
        return []
    rs = sorted(regions, key=lambda r: r["bin_lo"])
    clusters: list[dict] = [{"bin_lo": rs[0]["bin_lo"], "bin_hi": rs[0]["bin_hi"],
                             "regions": [rs[0]]}]
    for r in rs[1:]:
        c = clusters[-1]
        if (r["bin_lo"] - c["bin_hi"] - 1) <= max_gap_bins:
            c["bin_hi"] = r["bin_hi"]
            c["regions"].append(r)
        else:
            clusters.append({"bin_lo": r["bin_lo"], "bin_hi": r["bin_hi"], "regions": [r]})
    out = []
    for c in clusters:
        n_bins = int(c["bin_hi"] - c["bin_lo"] + 1)
        if n_bins < min_bins:
            continue
        peaks = [float(r["peak_db"]) for r in c["regions"]]
        sig_bins = int(sum(r["bin_hi"] - r["bin_lo"] + 1 for r in c["regions"]))
        best = max(c["regions"], key=lambda r: r["peak_db"])
        out.append({"bin_lo": int(c["bin_lo"]), "bin_hi": int(c["bin_hi"]),
                    "lo_hz": float(c["regions"][0]["lo_hz"]),
                    "hi_hz": float(c["regions"][-1]["hi_hz"]),
                    "bandwidth_hz": float(n_bins * df), "peak_db": float(max(peaks)),
                    "peak_freq_hz": float(best["peak_freq_hz"]),
                    "snr_over_floor_db": float(max(peaks) - min(peaks) + best["snr_over_floor_db"]),
                    "n_regions": len(c["regions"]), "n_signal_bins": sig_bins,
                    "occupancy": float(sig_bins / max(n_bins, 1)),
                    "regions": c["regions"], "n_bins": n_bins})
    out.sort(key=lambda c: -c["peak_db"])
    return out


# ---------------------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------------------
def analyse_spectrum(x: np.ndarray, fs: float, nperseg: int | None = None,
                     window: str = "hann", overlap: float = 0.5,
                     obw_frac: float = 0.99, thr_db: float = 4.0,
                     band_hz: tuple[float, float] | None = None,
                     reference_center_hz: float | None = None,
                     noise_density_hint: float | None = None,
                     noise_source: str | None = None) -> dict:
    """Full spectrum analysis of ``x``.

    ``band_hz`` restricts the analysis to a sub-band (used by the signal explorer after a
    region is selected).  ``reference_center_hz`` is the capture centre used to express the
    frequency offset; it defaults to 0 Hz (baseband capture).

    ``noise_density_hint`` supplies the noise power per Hz measured elsewhere.  This is used when
    the record being analysed is a *filtered copy* of a record that was already characterised: a
    band-pass filter leaves the in-band noise density unchanged, while the stopband attenuation
    makes the recorded spectrum look artificially quiet - measuring the floor again inside the
    filtered band would report an SNR tens of dB too high.
    """
    x = np.asarray(x)
    if x.ndim != 1 or x.size < 64:
        return {"ok": False, "message": f"insufficient samples for spectrum analysis (n={x.size})"}
    if np.allclose(np.abs(x), 0.0):
        return {"ok": False, "message": "the record is all zeros; there is nothing to analyse"}
    psd = welch_psd(x, fs, nperseg=nperseg, window=window, overlap=overlap)
    if not psd.get("ok"):
        return psd
    f_all, P_all = psd["freq_hz"], psd["psd"]
    df = psd["df_hz"]
    sel = np.ones(f_all.size, dtype=bool)
    if band_hz is not None:
        lo_b, hi_b = sorted([float(band_hz[0]), float(band_hz[1])])
        sel = (f_all >= lo_b) & (f_all <= hi_b)
        if np.count_nonzero(sel) < 16:
            return {"ok": False, "message": "the selected band is too narrow for a PSD estimate"}
    f = f_all[sel]
    P = P_all[sel]
    if P.size < 16:
        return {"ok": False, "message": "not enough PSD bins in the selected band"}
    nb = f.size
    ref_center = float(reference_center_hz if reference_center_hz is not None else 0.0)

    # ---- noise floor ---------------------------------------------------------------
    Pdb_raw = 10.0 * np.log10(np.maximum(P, 1e-30) * df)
    if noise_density_hint and noise_density_hint > 0:
        n0 = float(noise_density_hint)
        nf = {"ok": True, "noise_density": n0, "noise_density_db_hz": float(db(n0)),
              "noise_floor_db": float(db(n0 * df)), "df_hz": df,
              "signal_bin_fraction": 0.0, "reliable": True,
              "quantile_correction": 1.0, "n_segments": psd["n_segments"],
              "limitation": None,
              "method": noise_source or "noise density carried over from the source record "
                                        "(the segment is a filtered copy of it)"}
    else:
        nf = estimate_noise_density(f, P, df_hz=df, n_segments=psd["n_segments"])
        n0 = float(nf.get("noise_density", 0.0))
    floor_db = float(nf.get("noise_floor_db", float(np.median(Pdb_raw))))
    sm = _smooth_psd(Pdb_raw, 5)
    mad = float(np.median(np.abs(sm - np.median(sm))))
    sigma_db = 1.4826 * mad
    cf = cfar_threshold_db(P, n0, df, psd["n_segments"], p_fa=1e-3)
    peak_pk = float(np.max(Pdb_raw))
    dynamic_db = peak_pk - floor_db
    # A record whose floor sits 100 dB below its peak is not a measurement with a real noise
    # floor (float32 round-off or an exactly synthesised signal).  Say so instead of reporting a
    # -207 dB/Hz "floor" and an SNR of 190 dB.
    numerical_floor = bool(dynamic_db > 120.0 and floor_db < -120.0
                           and noise_density_hint is None)
    thr_ref = float(peak_pk - 100.0) if numerical_floor else floor_db
    floor_db_used = thr_ref
    thr = float(max(thr_db, cf["level_db"] - floor_db_used, 3.0))
    cfar_level_lin = n0 * df * 10 ** (thr_ref - floor_db) if numerical_floor else cf["level_linear"]
    mask_raw = P > cfar_level_lin                 # per-bin CFAR test (known false-alarm rate)
    mask = (sm > (floor_db_used + thr)) | mask_raw

    # ---- strongest region ----------------------------------------------------------
    regions = _find_regions(f, Pdb_raw, floor_db_used, thr)
    # a region must hold at least two adjacent bins to be treated as an emission
    regions = [r for r in regions if (r["bin_hi"] - r["bin_lo"] + 1) >= 2]
    # group neighbouring regions into emissions (AM sidebands, FSK tones, FM line combs)
    gap_bins = max(4, int(round(0.025 * nb)))
    emissions = cluster_regions(regions, df, max_gap_bins=gap_bins, min_bins=2)
    regions = [e for e in emissions]          # the headline list is now emission-level
    if not regions:
        # no emission above the floor: report the floor and mark everything else unavailable
        return {
            "ok": True, "no_signal": True, "freq_hz": f.tolist(), "psd_db": Pdb_raw.tolist(),
            "noise_floor_db": floor_db, "noise_reliable": bool(nf.get("reliable")),
            "numerical_floor": numerical_floor,
            "threshold_db": float(floor_db_used + thr), "fs": float(fs), "df_hz": float(df),
            "nperseg": psd["nperseg"], "window": window,
            "message": ("no emission stands out above the noise floor in this record "
                        f"(peak is {float(np.max(Pdb_raw)) - floor_db:.1f} dB above the estimated floor)"),
            "params": {
                "noise_floor": param("Noise floor", floor_db, "dB", status="ok",
                                     confidence=clamp01(1 - sigma_db / 6.0),
                                     method="median of the quietest 25 % of PSD bins",
                                     evidence=[f"PSD: Welch, {psd['nperseg']}-point {window}, "
                                               f"{psd['n_segments']} segments, {df:.1f} Hz/bin",
                                               f"threshold for signal presence = floor + {thr:.1f} dB "
                                               f"(4x smoothed-PSD MAD = {sigma_db:.2f} dB)"]),
                "snr": unavailable("SNR", "noise floor estimate only; no emission could be separated"),
                "occupied_bandwidth": unavailable("Occupied bandwidth",
                                                  "no emission detected above the noise floor"),
                "peak_frequency": unavailable("Peak frequency",
                                              "no emission detected above the noise floor"),
            },
        }

    r0 = emissions[0] if emissions else regions[0]
    lo_bin, hi_bin = r0["bin_lo"], r0["bin_hi"]
    seg_P = P[lo_bin:hi_bin + 1]
    seg_f = f[lo_bin:hi_bin + 1]
    # noise-corrected spectrum of the region
    Pcorr = np.maximum(seg_P - n0, 0.0)
    # the bins that pass the per-bin detection test carry the signal; bandwidth/centre statistics
    # are computed on those only, otherwise residual noise fluctuations inflate the 99 % bandwidth
    # of line-structured emissions (AM sidebands, FM line combs)
    sig_mask = seg_P > cfar_level_lin
    Pobw = Pcorr * sig_mask

    # occupied bandwidth on the noise-corrected spectrum
    # the occupied bandwidth counts every bin of the emission (an emission whose outer lines are
    # weak must not report a narrower band than the power it actually carries)
    obw = _cumulative_bandwidth(seg_f, Pcorr, obw_frac) if Pcorr.sum() > 0 else {"ok": False}
    if obw.get("ok"):
        obw_hz = float(obw["bandwidth_hz"])
        obw_lo, obw_hi = float(obw["lo_hz"]), float(obw["hi_hz"])
    else:
        obw_hz, obw_lo, obw_hi = r0["bandwidth_hz"], r0["lo_hz"], r0["hi_hz"]
    obw_raw = _cumulative_bandwidth(seg_f, seg_P, obw_frac)

    # power / SNR
    sig_power = float(np.sum(Pcorr) * df)                     # noise-corrected integrated power
    noise_in_band = float(n0 * max(obw_hz, df))
    snr_lin = sig_power / max(noise_in_band, 1e-300)
    snr_db = float(db(snr_lin))
    # in-band SNR measured against the whole detected region (independent cross-check)
    band_power = float(np.sum(seg_P) * df)
    noise_band = float(n0 * max(r0["bandwidth_hz"], df))
    snr_region_db = float(db(max(band_power - noise_band, 1e-300) / max(noise_band, 1e-300)))

    # peak & centre frequency
    ipk_sm = int(np.argmax(_smooth_psd(Pdb_raw, 5)[lo_bin:hi_bin + 1]))
    peak_f = float(seg_f[ipk_sm])
    if 0 < ipk_sm < seg_f.size - 1:
        try:
            peak_f = float(parabolic_peak(_smooth_psd(Pdb_raw, 5)[lo_bin:hi_bin + 1],
                                          seg_f, ipk_sm))
        except Exception:
            peak_f = float(seg_f[ipk_sm])
    Pcentre = Pobw if Pobw.sum() > 0 else Pcorr
    wsum = float(np.sum(Pcentre))
    centroid = float(np.sum(seg_f * Pcentre) / wsum) if wsum > 0 else float(np.mean(seg_f))
    edges = np.percentile(seg_f, [5, 95])
    edge_center = float(np.mean(edges))
    # power-weighted centroid and the 5/95 % edges normally agree; take the edge midpoint when
    # the spectrum is strongly asymmetric (skew > 0.25)
    skew = float(np.sum(Pcentre * (seg_f - centroid) ** 3) / max(wsum, 1e-300) /
                 max((np.sum(Pcentre * (seg_f - centroid) ** 2) / max(wsum, 1e-300)) ** 1.5, 1e-300))
    center_f = centroid if abs(skew) <= 0.25 else 0.5 * (centroid + edge_center)
    offset_hz = center_f - ref_center

    # A flat-topped emission (typical of QAM/PSK with a near-rectangular spectrum) has no
    # meaningful "peak frequency": report the centre frequency instead and say so.
    Pdb_sm = _smooth_psd(Pdb_raw, 5)[lo_bin:hi_bin + 1]
    flat_bw = _edge_bandwidth(seg_f, Pdb_sm, -0.5)
    flat_ratio = 0.0    # replaced below by the -3 dB / occupied-bandwidth ratio
    flat_top = bool(flat_ratio > 0.5 and obw_hz < 0.5 * (f[-1] - f[0]))
    line_like = bool(obw_hz <= max(4.0 * df, 0.0015 * float(fs)))
    if line_like:
        flat_top = False
    bw3 = _edge_bandwidth(seg_f, 10 * np.log10(np.maximum(seg_P, 1e-30) * df), -3.0)
    if obw_hz > 0 and bw3.get("ok"):
        flat_ratio = float(bw3["bandwidth_hz"] / obw_hz)
        flat_top = bool(flat_ratio > 0.55 and obw_hz < 0.5 * (f[-1] - f[0]))
        line_like = bool(obw_hz <= max(4.0 * df, 0.0015 * float(fs)))
        if line_like:
            flat_top = False
    bw20 = _edge_bandwidth(seg_f, 10 * np.log10(np.maximum(seg_P, 1e-30) * df), -20.0)
    peak_db = float(np.max(Pdb_raw[lo_bin:hi_bin + 1]))
    peak_to_floor = peak_db - floor_db

    # confidence of the headline numbers
    snr_conf = clamp01((snr_db - 2.0) / 18.0)
    floor_conf = clamp01(1.0 - max(0.0, nf["signal_bin_fraction"] - 0.25) / 0.5) * \
        clamp01(1.0 - sigma_db / 8.0)
    if not nf.get("reliable") or numerical_floor:
        floor_conf = min(floor_conf, 0.35)
    obw_conf = clamp01(0.45 + 0.4 * snr_conf + 0.15 * clamp01(obw_hz / max(r0["bandwidth_hz"], df)))

    occupancy = float(np.count_nonzero(mask) / nb)
    evidence_common = [
        f"Welch PSD: {psd['nperseg']}-point {window} window, {psd['n_segments']} averaged segments, "
        f"{df:.1f} Hz/bin, {P.size} bins analysed",
        f"noise floor = {floor_db:.2f} dB per bin ({nf['method']}); smoothed-floor spread "
        f"{sigma_db:.2f} dB",
        (f"the floor is {dynamic_db:.0f} dB below the peak: this record is effectively "
         f"noise-free (float32 round-off or an exactly synthesised signal), so the SNR reported "
         f"below is a lower bound and the detection threshold was set 100 dB below the peak"
         if numerical_floor else
         f"detection threshold = floor + {thr:.1f} dB ({cf['method']}; every bin is also tested "
         f"individually against the same level)"),
        f"detected emission: {obw_lo:,.0f} .. {obw_hi:,.0f} Hz "
        f"({human_si(obw_hi - obw_lo, 'Hz')} wide), peak {peak_db:.1f} dB/Hz "
        f"({peak_to_floor:.1f} dB above the floor)",
        f"signal-occupied fraction of the analysed band = {occupancy * 100:.1f} %",
    ]
    limitations = []
    if numerical_floor:
        limitations.append("the record appears to be noise-free (the estimated floor is "
                           f"{dynamic_db:.0f} dB below the peak, i.e. at the numerical limit): "
                           "the noise floor and SNR are lower bounds, not measurements")
    if not nf.get("reliable"):
        limitations.append(str(nf.get("limitation")))
    if abs(offset_hz) < 2 * df:
        limitations.append("the detected emission is centred on 0 Hz (capture centre); a DC spur "
                           "or local-oscillator leakage can produce the same signature")
    if occupancy > 0.6:
        limitations.append("the analysed record is more than 60 % occupied by signal, so noise "
                           "floor, SNR and bandwidth are less certain")
    if peak_to_floor < 6:
        limitations.append("the emission is only weakly above the floor; SNR is a rough estimate")
    if r0.get("n_regions", 1) > 3:
        limitations.append(f"the emission has a multi-line structure ({r0['n_regions']} spectral "
                           "regions within "
                           f"{human_si(gap_bins * df, 'Hz')} of each other: AM sidebands, FSK tones "
                           "or an FM line comb); the bandwidth spans the detected lines")
    if r0.get("occupancy", 1.0) < 0.5 and r0.get("n_regions", 1) > 1:
        limitations.append("less than half of the reported band carries detected power: the "
                           "signal may be pulsed or the band may include a second emission")

    params = {
        "noise_floor": param("Noise floor", round(floor_db, 2), "dB/Hz", status="ok",
                             confidence=round(floor_conf, 2), method=nf["method"],
                             evidence=evidence_common[:4],
                             limitations=[] if nf.get("reliable") else [nf.get("limitation")]),
        "peak_frequency": param("Peak frequency", round(peak_f, 1), "Hz",
                                status="low_confidence" if flat_top else "ok",
                                confidence=round(clamp01(snr_conf + 0.1 * min(1.0, peak_to_floor / 20)), 2),
                                method="smoothed PSD maximum with parabolic interpolation",
                                evidence=([f"peak level {peak_db:.1f} dB/Hz, {peak_to_floor:.1f} dB above floor",
                                           f"FFT resolution {df:.1f} Hz/bin"] if not flat_top else
                                          [f"the emission is flat-topped: the -0.5 dB bandwidth is "
                                           f"{human_si(flat_bw.get('bandwidth_hz') or 0, 'Hz')} of the "
                                           f"{human_si(obw_hz, 'Hz')} occupied bandwidth "
                                           f"({flat_ratio * 100:.0f} %)"]),
                                limitations=(["the emission is spectrally flat over most of its "
                                              "bandwidth (typical of a modulated carrier), so the peak "
                                              "bin is not a carrier estimate - use the centre frequency"]
                                             if flat_top else [])),
        "center_frequency": param("Centre frequency (emission)", round(center_f, 1), "Hz", status="ok",
                                  confidence=round(clamp01(min(snr_conf + 0.15, 0.98) * (1 - min(abs(skew), 1) * 0.2)), 2),
                                  method="power-weighted spectral centroid of the detected emission"
                                         + (" (blended with the 5/95 % band edges: asymmetric spectrum)"
                                            if abs(skew) > 0.25 else ""),
                                  evidence=[f"centroid {centroid:,.1f} Hz, 5/95 % edge centre {edge_center:,.1f} Hz, "
                                            f"skew {skew:+.2f}",
                                            f"peak at {peak_f:,.1f} Hz"]),
        "frequency_offset": param("Frequency offset from capture centre", round(offset_hz, 1), "Hz",
                                  status="ok" if snr_db > 3 else "low_confidence",
                                  confidence=round(clamp01(snr_conf + 0.1), 2),
                                  method="centre frequency minus reference centre "
                                         f"({ref_center:,.0f} Hz)",
                                  evidence=[f"offset = {center_f:,.1f} Hz - {ref_center:,.0f} Hz"],
                                  limitations=([] if abs(offset_hz) > 2 * df else
                                               ["offset is within two PSD bins of the centre; this may be "
                                                "capture-centre leakage rather than a true offset"])),
        "occupied_bandwidth": param(f"Occupied bandwidth ({int(obw_frac * 100)} %)",
                                    round(obw_hz, 1), "Hz", status="ok",
                                    confidence=round(obw_conf, 2),
                                    method=f"narrowest band containing {int(obw_frac * 100)} % of the "
                                           "noise-corrected power, grown outward from the peak",
                                    evidence=[f"band {obw_lo:,.1f} .. {obw_hi:,.1f} Hz",
                                              f"uncorrected {int(obw_frac * 100)} % bandwidth "
                                              f"{obw_raw.get('bandwidth_hz', float('nan')):,.1f} Hz "
                                              "(inflation is the noise inside the band)",
                                              f"analysis resolution {df:.1f} Hz/bin"],
                                    limitations=([f"resolution {df:.1f} Hz/bin limits the accuracy to "
                                                   f"about +/-{df:.0f} Hz"] if obw_hz < 10 * df else [])),
        "snr": param("SNR (in the occupied band)", round(snr_db, 2), "dB",
                     status="low_confidence" if numerical_floor else "ok",
                     confidence=round(min(snr_conf, 0.4) if numerical_floor else snr_conf, 2),
                     method="10*log10(signal power / noise power), both integrated over the "
                            f"{int(obw_frac * 100)} % occupied band",
                     evidence=[f"signal power {db(sig_power):.2f} dB (noise-corrected integral)",
                               f"noise density {floor_db:.2f} dB/Hz over {human_si(obw_hz, 'Hz')}",
                               f"cross-check (whole detected region) {snr_region_db:.2f} dB"],
                     limitations=limitations),
        "bandwidth_3db": param("Bandwidth (-3 dB)", round(bw3.get("bandwidth_hz", float("nan")), 1),
                               "Hz", status="ok" if bw3.get("ok") else "unable",
                               confidence=round(clamp01(snr_conf * 0.9), 2),
                               method="level crossing 3 dB below the peak of the smoothed PSD",
                               evidence=[f"analysis resolution {df:.1f} Hz/bin"]),
        "bandwidth_20db": param("Bandwidth (-20 dB)", round(bw20.get("bandwidth_hz", float("nan")), 1),
                                "Hz", status="ok" if bw20.get("ok") else "unable",
                                confidence=round(clamp01(snr_conf * 0.8), 2),
                                method="level crossing 20 dB below the peak of the smoothed PSD"),
    }

    # secondary emissions (other signals in the same record) - passed to the detector, which
    # does its own time-resolved work; here we simply report them.
    others = [{k: v for k, v in r.items() if k != "regions"} for r in emissions[1:8]]

    return {
        "ok": True, "no_signal": False, "freq_hz": f.tolist(), "psd_db": Pdb_raw.tolist(),
        "psd": P.tolist(), "df_hz": float(df), "fs": float(fs), "nperseg": psd["nperseg"],
        "window": window, "n_segments": psd["n_segments"], "estimator": psd["estimator"],
        "noise_floor_db": floor_db, "noise_density": n0,
        "noise_source": (noise_source if noise_density_hint else "estimated from this record"),
        "noise_reliable": bool(nf.get("reliable")),
        "threshold_db": float(floor_db_used + thr), "cfar": cf, "peak_db": peak_db,
        "dynamic_range_db": float(dynamic_db), "numerical_floor": numerical_floor, "peak_to_floor_db": peak_to_floor,
        "peak_frequency_hz": peak_f, "center_frequency_hz": center_f, "centroid_hz": centroid,
        "frequency_offset_hz": offset_hz, "reference_center_hz": ref_center,
        "obw_99_hz": obw_hz, "obw_lo_hz": obw_lo, "obw_hi_hz": obw_hi,
        "obw_raw_hz": float(obw_raw.get("bandwidth_hz", float("nan"))) if obw_raw.get("ok") else None,
        "bw_3db_hz": bw3.get("bandwidth_hz"), "bw_20db_hz": bw20.get("bandwidth_hz"),
        "flat_top": flat_top, "flat_top_ratio": float(flat_ratio), "line_like": line_like,
        "bw3_over_obw": float((bw3.get("bandwidth_hz") or 0.0) / obw_hz) if obw_hz > 0 else None,
        "snr_db": snr_db, "snr_region_db": snr_region_db, "signal_power_db": float(db(sig_power)),
        "noise_power_db": float(db(noise_in_band)), "occupied_fraction": occupancy,
        "region": {k: v for k, v in r0.items() if k != "regions"},
        "emissions": [{k: v for k, v in r.items() if k != "regions"} for r in emissions],
        "other_regions": others, "skew": skew, "cluster_gap_bins": int(gap_bins),
        "params": params,
        "evidence": evidence_common, "limitations": limitations,
    }


def band_snr(x: np.ndarray, fs: float, f_lo: float, f_hi: float,
             nperseg: int | None = None) -> dict:
    """SNR of anything inside ``f_lo..f_hi`` using a noise floor from the same record."""
    a = analyse_spectrum(x, fs, nperseg=nperseg, band_hz=(f_lo, f_hi))
    if not a.get("ok"):
        return a
    return {"ok": True, "snr_db": a["snr_db"], "noise_floor_db": a["noise_floor_db"],
            "signal_power_db": a["signal_power_db"], "obw_hz": a["obw_99_hz"],
            "center_frequency_hz": a["center_frequency_hz"], "no_signal": a.get("no_signal", False)}


def band_power_db(x: np.ndarray, fs: float, f_lo: float, f_hi: float,
                  nperseg: int | None = None) -> float | None:
    """Integrated power in dB inside a band (no noise correction)."""
    psd = welch_psd(x, fs, nperseg=nperseg)
    if not psd.get("ok"):
        return None
    f, P = psd["freq_hz"], psd["psd"]
    m = (f >= f_lo) & (f <= f_hi)
    if not np.any(m):
        return None
    return float(db(np.sum(P[m]) * psd["df_hz"]))


def envelope_spectrum(x: np.ndarray, fs: float, nperseg: int | None = None) -> dict:
    """Spectrum of |x| (after DC removal): the AM signature lives here."""
    env = np.abs(np.asarray(x, dtype=np.complex128 if np.iscomplexobj(x) else np.float64))
    env = env - np.mean(env)
    if env.size < 64:
        return {"ok": False, "message": "insufficient samples for an envelope spectrum"}
    psd = welch_psd(env, fs, nperseg=nperseg)
    if not psd.get("ok"):
        return psd
    f, P = psd["freq_hz"], psd["psd"]
    Pdb = 10 * np.log10(np.maximum(P, 1e-30) * psd["df_hz"])
    m = f > 20.0
    if not np.any(m):
        return {"ok": False, "message": "no positive-frequency envelope bins"}
    ipk = int(np.argmax(Pdb[m]))
    fp = f[m][ipk]
    lvl = float(Pdb[m][ipk])
    med = float(np.median(Pdb[m]))
    return {"ok": True, "freq_hz": f.tolist(), "psd_db": Pdb.tolist(),
            "modulation_freq_hz": float(fp), "tone_peak_db": lvl, "floor_db": med,
            "tone_prominence_db": float(lvl - med), "df_hz": psd["df_hz"],
            "method": "Welch PSD of the envelope |x| with the mean removed"}


def estimate_am_index(x: np.ndarray, fs: float) -> dict:
    """Amplitude-modulation index from the envelope statistics."""
    env = np.abs(np.asarray(x, dtype=np.complex128 if np.iscomplexobj(x) else np.float64))
    if env.size < 64:
        return {"ok": False, "message": "insufficient samples"}
    mu = float(np.mean(env))
    if mu <= 0:
        return {"ok": False, "message": "zero mean envelope"}
    lo, hi = np.percentile(env, [2.0, 98.0])
    m = float((hi - lo) / (hi + lo)) if (hi + lo) > 0 else 0.0
    es = envelope_spectrum(x, fs)
    tone = es.get("modulation_freq_hz") if es.get("ok") else None
    prom = es.get("tone_prominence_db") if es.get("ok") else None
    depth = float(np.std(env) / mu)
    return {"ok": True, "modulation_index": float(np.clip(m, 0.0, 1.0)), "envelope_cv": depth,
            "modulation_freq_hz": tone, "tone_prominence_db": prom,
            "method": "2/98 percentile envelope contrast m=(max-min)/(max+min), cross-checked "
                      "with the envelope spectrum",
            "confidence": clamp01(0.25 + 0.5 * clamp01((prom or 0) / 20.0) + 0.25 * clamp01(depth))}


def instantaneous_frequency(x: np.ndarray, fs: float, smooth: int = 3,
                            remove_mean: bool = True) -> dict:
    """Instantaneous frequency (Hz) of an analytic signal, plus its statistics."""
    x = np.asarray(x)
    if not np.iscomplexobj(x):
        x = sigproc.hilbert(x)
    if x.size < 32:
        return {"ok": False, "message": "insufficient samples"}
    ph = np.unwrap(np.angle(x))
    dphi = np.diff(ph)
    # remove a linear phase drift (carrier offset) so the statistics describe the modulation
    if remove_mean and dphi.size > 16:
        t = np.arange(dphi.size, dtype=np.float64)
        try:
            slope = float(np.polyfit(t, dphi, 1)[0])
        except Exception:
            slope = 0.0
        dphi = dphi - slope * t
    inst = dphi * fs / (2.0 * np.pi)
    if smooth > 1:
        kernel = np.ones(int(smooth)) / float(smooth)
        inst = np.convolve(inst, kernel, mode="same")
    if inst.size < 8:
        return {"ok": False, "message": "insufficient samples for an IF estimate"}
    mean_hz = float(np.mean(inst))
    rms_dev = float(np.sqrt(np.mean((inst - mean_hz) ** 2)))
    return {"ok": True, "inst_freq_hz": inst.tolist(), "sf_hz": float(np.std(inst)),
            # aliases used by the FSK demodulator: the constant part of the estimate is the
            # residual carrier, the fluctuation about it is the deviation
            "residual_carrier_hz": mean_hz, "rms_deviation_hz": rms_dev,
            "mean_hz": mean_hz,
            "max_abs_hz": float(np.max(np.abs(inst))),
            "percentiles_hz": {p: float(v) for p, v in
                               zip((5, 25, 50, 75, 95), np.percentile(np.abs(inst), [5, 25, 50, 75, 95]))},
            "method": "phase-difference instantaneous frequency, linear phase drift removed, "
                      f"{int(smooth)}-sample moving average"}


def region_spectrum_params(x: np.ndarray, fs: float, f_lo: float, f_hi: float,
                           reference_center_hz: float = 0.0) -> dict:
    """Spectrum parameters restricted to one band (used per detected signal)."""
    return analyse_spectrum(x, fs, band_hz=(f_lo, f_hi), reference_center_hz=reference_center_hz)
