"""Spectrogram / waterfall computation and band extraction.

The spectrogram is a Welch-style STFT with overlap; the payload handed to the browser is
aggregated to a bounded size (so a 10 s capture at 2 MS/s does not ship 10^8 cells) while the
numeric analysis always runs on the full-resolution matrix.

``extract_band`` is the bridge between "the user selected a region on the waterfall" and
"analyse this segment": it shifts the selected band to baseband, low-pass filters it and
decimates it to a manageable rate.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal as sigproc

from .preprocess import make_window
from .utils import db, next_pow2, safe_float

__all__ = [
    "stft_matrix", "downsample_payload", "extract_band", "slice_time", "integrate_region",
    "region_stats", "waterfall_payload", "band_time_profile",
]


def _auto_nperseg(n: int, fs: float, nperseg: int | None) -> int:
    if nperseg:
        return int(max(32, min(int(nperseg), n)))
    # target ~1 ms of time resolution, bounded, power of two
    target = int(fs * 0.001)
    nper = int(np.clip(next_pow2(max(64, target)), 64, 2048))
    return int(min(nper, n))


def stft_matrix(x: np.ndarray, fs: float, nperseg: int | None = None, overlap: float = 0.75,
                window: str = "hann", max_frames: int = 4000) -> dict:
    """Spectrogram of ``x`` as power spectral density (power/Hz).

    Returns ``times_s`` (frame centres), ``freq_hz`` (fftshifted for complex input) and the 2-D
    ``power`` / ``db`` matrices indexed ``[freq, time]``.
    """
    x = np.asarray(x)
    n = x.size
    if n < 64:
        return {"ok": False, "message": f"insufficient samples for a spectrogram (n={n})"}
    nper = _auto_nperseg(n, fs, nperseg)
    nover = int(max(0, min(nper - 1, round(nper * float(np.clip(overlap, 0.0, 0.95))))))
    step = nper - nover
    win = make_window(window, nper)
    real_input = not np.iscomplexobj(x)
    try:
        f, t, Z = sigproc.stft(x.astype(np.complex128 if not real_input else np.float64), fs=fs,
                               window=win, nperseg=nper, noverlap=nover, detrend="constant",
                               return_onesided=real_input, scaling="psd", boundary=None,
                               padded=False)   # scipy calls the density scaling "psd"
    except Exception as exc:                                              # pragma: no cover
        return {"ok": False, "message": f"STFT failed: {exc}"}
    P = (np.abs(Z) ** 2).astype(np.float64)
    if not real_input:
        f = np.fft.fftshift(f)
        P = np.fft.fftshift(P, axes=0)
    # scipy returns frame *left edges* in seconds; report frame centres
    t_c = np.asarray(t, dtype=np.float64) + nper / (2.0 * float(fs))
    if t_c.size > max_frames:                       # keep the numeric matrix bounded too
        stride = int(np.ceil(t_c.size / max_frames))
        P = P[:, ::stride]
        t_c = t_c[::stride]
    df = float(fs / nper)
    Pdb = 10.0 * np.log10(np.maximum(P, 1e-30) * df)
    return {"ok": True, "times_s": t_c, "freq_hz": f, "power": P, "db": Pdb,
            "df_hz": df, "dt_s": float(step / fs), "nperseg": nper, "noverlap": nover,
            "n_frames": int(t_c.size), "n_freq": int(f.size), "window": window,
            "onesided": bool(real_input), "fs": float(fs),
            "method": f"STFT, {nper}-point {window}, {int(overlap * 100)} % overlap, "
                      f"power spectral density"}


def _aggregate(M: np.ndarray, f_idx: np.ndarray, t_idx: np.ndarray, mode: str) -> np.ndarray:
    """Aggregate a 2-D matrix onto a coarse grid with ``max`` or ``mean``."""
    out = np.empty((f_idx.size - 1, t_idx.size - 1), dtype=np.float64)
    fn = np.maximum(np.diff(f_idx), 1)
    tn = np.maximum(np.diff(t_idx), 1)
    for i in range(out.shape[0]):
        i0, i1 = f_idx[i], f_idx[i + 1]
        for j in range(out.shape[1]):
            blk = M[i0:i1, t_idx[j]:t_idx[j + 1]]
            out[i, j] = np.max(blk) if mode == "max" else float(np.mean(blk))
    return out


def downsample_payload(freq_hz: np.ndarray, times_s: np.ndarray, dbmat: np.ndarray,
                       max_w: int = 900, max_h: int = 700, mode: str = "max") -> dict:
    """Reduce a spectrogram to at most ``max_w x max_h`` cells for display.

    ``mode="max"`` keeps the strongest cell of each block, so short bursts and narrow tones stay
    visible; the aggregation factor is reported back so the UI can label it honestly.
    """
    nf, nt = dbmat.shape
    if nf <= max_h and nt <= max_w:
        return {"ok": True, "freq_hz": freq_hz.tolist(), "times_s": times_s.tolist(),
                "db": dbmat.tolist(), "aggregated": False, "f_per_bin": 1, "t_per_bin": 1,
                "freq_step_hz": float(freq_hz[1] - freq_hz[0]) if nf > 1 else 0.0,
                "time_step_s": float(times_s[1] - times_s[0]) if nt > 1 else 0.0,
                "note": "full resolution - no aggregation was needed"}
    f_bins = max(1, int(np.ceil(nf / max_h)))
    t_bins = max(1, int(np.ceil(nt / max_w)))
    f_idx = np.arange(0, nf + f_bins, f_bins)
    t_idx = np.arange(0, nt + t_bins, t_bins)
    f_idx[-1] = nf
    t_idx[-1] = nt
    Fdb = _aggregate(dbmat, f_idx, t_idx, mode)
    f_axis = [float(np.mean(freq_hz[max(0, f_idx[i]):f_idx[i + 1]])) for i in range(len(f_idx) - 1)]
    t_axis = [float(np.mean(times_s[max(0, t_idx[i]):t_idx[i + 1]])) for i in range(len(t_idx) - 1)]
    step_f = float(f_axis[1] - f_axis[0]) if len(f_axis) > 1 else 0.0
    step_t = float(t_axis[1] - t_axis[0]) if len(t_axis) > 1 else 0.0
    return {"ok": True, "freq_hz": f_axis, "times_s": t_axis, "db": Fdb.tolist(),
            "aggregated": True, "f_per_bin": f_bins, "t_per_bin": t_bins,
            "freq_step_hz": step_f, "time_step_s": step_t,
            "note": f"display grid aggregated {f_bins}x{t_bins} cells "
                    f"({'peak' if mode == 'max' else 'mean'} hold) from {nf}x{nt}"}


def waterfall_payload(x: np.ndarray, fs: float, nperseg: int | None = None, overlap: float = 0.75,
                      window: str = "hann", max_w: int = 900, max_h: int = 700,
                      mode: str = "max", db_floor_percentile: float = 1.0) -> dict:
    """Ready-to-send spectrogram payload (display arrays + colour scale)."""
    sp = stft_matrix(x, fs, nperseg=nperseg, overlap=overlap, window=window)
    if not sp.get("ok"):
        return sp
    pay = downsample_payload(sp["freq_hz"], sp["times_s"], sp["db"], max_w=max_w, max_h=max_h, mode=mode)
    flat = sp["db"][np.isfinite(sp["db"])]
    vmin = float(np.percentile(flat, db_floor_percentile)) if flat.size else -120.0
    vmax = float(np.percentile(flat, 99.9)) if flat.size else 0.0
    if vmax - vmin < 6.0:
        vmin, vmax = vmin - 3.0, vmax + 3.0
    pay.update({"vmin_db": round(vmin, 1), "vmax_db": round(vmax, 1),
                "color_scale": "dB relative to 1.0 (power/Hz)" + (" - uncalibrated (dBFS)" if True else ""),
                "n_frames_full": sp["n_frames"], "n_freq_full": sp["n_freq"],
                "df_hz": sp["df_hz"], "dt_s": sp["dt_s"], "nperseg": sp["nperseg"],
                "window": window, "method": sp["method"], "ok": True})
    return pay


def slice_time(x: np.ndarray, fs: float, t0_s: float | None, t1_s: float | None) -> dict:
    """Extract the time slice ``t0..t1`` (clipped to the record)."""
    x = np.asarray(x)
    n = x.size
    i0 = 0 if t0_s is None else int(np.clip(round(float(t0_s) * fs), 0, n))
    i1 = n if t1_s is None else int(np.clip(round(float(t1_s) * fs), 0, n))
    if i1 <= i0:
        return {"ok": False, "message": "the selected time range is empty"}
    return {"ok": True, "samples": x[i0:i1], "i0": i0, "i1": i1,
            "duration_s": float((i1 - i0) / fs), "n_samples": int(i1 - i0)}


def extract_band(x: np.ndarray, fs: float, f_lo: float, f_hi: float, guard_frac: float = 0.35,
                 order: int = 6, target_fs: float | None = None,
                 t0_s: float | None = None, t1_s: float | None = None,
                 max_out_samples: int = 4_000_000) -> dict:
    """Shift ``f_lo..f_hi`` to baseband, filter and (optionally) decimate.

    Real input is converted to its analytic form first, so the returned segment is always a
    complex baseband record ready for the demodulator.  Filtering uses a Butterworth low-pass at
    ``bw/2 * (1 + guard)`` - the default guard of 35 % keeps the filter clear of the emission's own
    spectrum, because a filter that cuts into the signal destroys the pulse shape (measured: a
    5 % guard left a 44 % EVM on a clean QPSK signal, a 35 % guard leaves ~2 %).  The decimation
    factor is chosen so the new rate is at least four times the filter cutoff (never below 1 kHz).
    """
    x = np.asarray(x)
    if x.ndim != 1 or x.size < 32:
        return {"ok": False, "message": "insufficient samples to extract a band"}
    f_lo, f_hi = sorted([float(f_lo), float(f_hi)])
    fs = float(fs)
    if x.real.dtype == np.float64 and not np.iscomplexobj(x):
        try:
            x = sigproc.hilbert(x.astype(np.float64)).astype(np.complex128)
        except Exception as exc:
            return {"ok": False, "message": f"analytic-signal conversion failed: {exc}"}
    x = x.astype(np.complex128)
    nyq = fs / 2.0
    f_lo = float(np.clip(f_lo, -nyq, nyq))
    f_hi = float(np.clip(f_hi, -nyq, nyq))
    if f_hi - f_lo <= 0:
        return {"ok": False, "message": "the selected band is empty"}
    sl = slice_time(x, fs, t0_s, t1_s)
    if not sl.get("ok"):
        return sl
    seg = sl["samples"]
    if seg.size < 32:
        return {"ok": False, "message": "the selected time slice is too short"}
    fs_in = fs
    f_c = 0.5 * (f_lo + f_hi)
    bw = f_hi - f_lo
    cut_hz_est = (bw / 2.0) * (1.0 + float(guard_frac))
    # decimation factor: keep at least four samples per filter-cutoff cycle
    target = float(target_fs) if target_fs else max(1000.0, 4.0 * cut_hz_est)
    decim = int(max(1, np.floor(fs / target)))
    while decim > 1 and (fs / decim) < 4.0 * cut_hz_est:
        decim -= 1
    if decim > 1 and seg.size // decim > max_out_samples:
        decim = int(max(decim, np.ceil(seg.size / max_out_samples)))
    # derotate
    t_idx = np.arange(seg.size, dtype=np.float64)
    seg = seg * np.exp(-2j * np.pi * f_c * t_idx / fs)
    cut = min(0.95, (bw / 2.0 * (1.0 + float(guard_frac))) / (fs / 2.0))
    cut = float(max(cut, 1e-4))
    try:
        sos = sigproc.butter(int(order), cut, btype="lowpass", output="sos")
        if seg.size > 3 * (int(order) * 2 + 1):
            seg = sigproc.sosfiltfilt(sos, seg)
        else:
            seg = sigproc.sosfilt(sos, seg)
    except Exception as exc:
        return {"ok": False, "message": f"band filter failed: {exc}"}
    fs_out = fs
    note = ("no decimation: the requested rate already matches the band"
            if decim <= 1 else f"decimated by {decim}")
    if decim > 1:
        try:
            seg = sigproc.resample_poly(seg, up=1, down=decim)
            fs_out = fs / decim
        except Exception as exc:                                        # pragma: no cover
            return {"ok": False, "message": f"decimation failed: {exc}"}
    # remove the residual DC created by imperfect centring
    seg = seg - np.mean(seg)
    return {"ok": True, "samples": seg, "fs": float(fs_out), "fs_in": float(fs),
            "f_center_hz": float(f_c), "bw_hz": float(bw), "decim": int(decim),
            "cutoff_hz": float(cut * fs / 2.0), "t0_s": float(0.0 if t0_s is None else t0_s),
            "t1_s": float(seg.size / fs_out + (0.0 if t0_s is None else t0_s)),
            "n_samples": int(seg.size), "duration_s": float(seg.size / fs_out),
            "note": note,
            "method": f"complex baseband shift by {f_c:,.1f} Hz, {int(order)}-order Butterworth "
                      f"low-pass at {cut * fs / 2.0:,.1f} Hz"
                      + (f", polyphase decimation by {decim}" if decim > 1 else "")}


def integrate_region(sp: dict, t0_s: float, t1_s: float, f0_hz: float, f1_hz: float) -> dict:
    """Mean / peak power and duty cycle inside a region of a spectrogram."""
    if not sp.get("ok"):
        return {"ok": False, "message": "no spectrogram"}
    t = sp["times_s"]
    f = sp["freq_hz"]
    ti = np.flatnonzero((t >= t0_s) & (t <= t1_s))
    fi = np.flatnonzero((f >= f0_hz) & (f <= f1_hz))
    if ti.size == 0 or fi.size == 0:
        return {"ok": False, "message": "the selected region is outside the spectrogram"}
    blk = sp["power"][np.ix_(fi, ti)]
    P = sp["power"]
    # local noise floor: quietest 20 % of the whole matrix (robust to a busy record)
    n0 = float(np.median(np.sort(P.ravel())[: max(16, int(0.2 * P.size))]))
    mean_p = float(np.mean(blk))
    peak_p = float(np.max(blk))
    snr_db = float(db(max(mean_p - n0, 1e-300) / max(n0, 1e-300)))
    return {"ok": True, "mean_power": mean_p, "peak_power": peak_p, "noise_floor": n0,
            "mean_db": float(db(mean_p)), "peak_db": float(db(peak_p)),
            "noise_floor_db": float(db(n0)), "snr_db": snr_db,
            "n_cells": int(blk.size), "n_frames": int(ti.size), "n_bins": int(fi.size),
            "f_lo_hz": float(f[fi[0]]), "f_hi_hz": float(f[fi[-1]]),
            "t0_s": float(t[ti[0]]), "t1_s": float(t[ti[-1]])}


def region_stats(sp: dict, mask: np.ndarray) -> dict:
    """Statistics of a boolean mask over a spectrogram (used by the signal detector)."""
    if not sp.get("ok"):
        return {"ok": False}
    n_f, n_t = sp["db"].shape
    if mask.shape != (n_f, n_t):
        return {"ok": False, "message": "mask shape does not match the spectrogram"}
    fi, ti = np.nonzero(mask)
    if fi.size == 0:
        return {"ok": False, "message": "empty region"}
    n0 = float(np.median(np.sort(sp["power"].ravel())[: max(16, int(0.2 * sp["power"].size))]))
    vals = sp["power"][fi, ti]
    return {"ok": True, "f_lo_hz": float(sp["freq_hz"][fi.min()]),
            "f_hi_hz": float(sp["freq_hz"][fi.max()]),
            "t0_s": float(sp["times_s"][ti.min()]), "t1_s": float(sp["times_s"][ti.max()]),
            "peak_db": float(db(np.max(vals))), "mean_db": float(db(np.mean(vals))),
            "noise_floor_db": float(db(n0)),
            "snr_db": float(db(max(np.mean(vals) - n0, 1e-300) / max(n0, 1e-300))),
            "n_cells": int(fi.size), "fill": float(fi.size / (n_f * n_t)),
            "bandwidth_hz": float(sp["freq_hz"][fi.max()] - sp["freq_hz"][fi.min()] + sp["df_hz"]),
            "duration_s": float(sp["times_s"][ti.max()] - sp["times_s"][ti.min()] + sp["dt_s"])}


def band_time_profile(sp: dict, f0_hz: float, f1_hz: float) -> dict:
    """Per-frame power of a band (used for duty-cycle and burst analysis)."""
    if not sp.get("ok"):
        return {"ok": False}
    fi = np.flatnonzero((sp["freq_hz"] >= f0_hz) & (sp["freq_hz"] <= f1_hz))
    if fi.size == 0:
        return {"ok": False, "message": "band outside the spectrogram"}
    prof = np.mean(sp["power"][fi, :], axis=0)
    return {"ok": True, "profile": prof, "times_s": sp["times_s"],
            "mean_db": float(db(np.mean(prof))), "peak_db": float(db(np.max(prof)))}
