"""Hybrid automatic modulation classification.

Three independent evidence sources are combined, and their individual votes are
kept in the output so the decision can be inspected (SIH §9/§21):

1. **DSP feature rules** - classical statistical features with explicit, documented
   thresholds (envelope constancy, instantaneous-frequency variation, phase
   clustering after de-rotation, envelope-spectrum tones, carrier-line presence);
2. **Constellation evidence** - each candidate modulation is actually demodulated
   (blind carrier + timing recovery) and scored by the *measured* EVM of the
   resulting constellation;
3. **Optional classical ML** - a scikit-learn model trained on synthetic data with
   the same feature vector, used as a tie-breaker with a bounded weight.

If no model file is present the classifier reports that it is running DSP-only; it
never silently substitutes a guess for a measurement.
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
from scipy import signal as sigproc
from scipy.stats import kurtosis as _sp_kurtosis

from . import demod as dm
from . import synth
from .preprocess import make_window
from .utils import param, clamp01, STATUS_OK, STATUS_LOW, STATUS_UNAVAILABLE

LINEAR = ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM")
ALL_MODS = LINEAR + ("2FSK", "GFSK", "AM", "FM")

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "ml", "models", "modclass.joblib")

FEATURE_NAMES = [
    "sigma_aa", "envelope_ripple_db", "gamma_max", "sigma_ap", "sigma_dp", "sigma_af",
    "mag_kurtosis", "spectral_flatness", "dc_carrier_line_db", "envelope_tone_db",
    "phase_cluster_2", "phase_cluster_4", "phase_cluster_8", "if_bimodality",
    "if_constancy_lp", "if_bw_over_obw", "obw_over_rs", "phase_cluster_16",
    "phase_cluster_2_blk", "phase_cluster_4_blk", "phase_cluster_8_blk", "if_edge_sharpness",
]


# --------------------------------------------------------------------------- #
#  Feature extraction
# --------------------------------------------------------------------------- #
def extract_features(x: np.ndarray, fs: float, rs: float | None = None,
                     obw_hz: float | None = None) -> dict:
    """Classical AMC feature vector, all quantities measured from the samples."""
    x = np.asarray(x, dtype=np.complex128)
    n = x.size
    if n < 256:
        return {"ok": False, "error": "fewer than 256 samples"}
    mag = np.abs(x)
    mean_mag = float(np.mean(mag))
    if mean_mag <= 0:
        return {"ok": False, "error": "zero-power signal"}
    sigma_aa = float(np.std(mag) / mean_mag)

    # gamma_max: normalised peak of the |x|^2 spectrum (standard AMC feature)
    p = mag ** 2
    p_c = p - np.mean(p)
    nper = int(min(8192, max(128, n // 4)))
    f, P = sigproc.welch(p_c, fs=fs, nperseg=nper, window=make_window("hann", nper),
                         return_onesided=False, scaling="density")
    o = np.argsort(f)
    f, P = f[o], P[o]
    gamma_max = float(np.max(P) / max(np.mean(P), 1e-30))
    gamma_max_db = 10 * math.log10(max(gamma_max, 1e-30))

    # phase features (amplitude-gated)
    gate = mag >= 0.35 * np.max(mag)
    if np.count_nonzero(gate) < 32:
        gate = np.ones_like(mag, dtype=bool)
    ph_gated = np.angle(x[gate])
    ph_gated = ph_gated - np.angle(np.mean(np.exp(1j * ph_gated)))
    sigma_ap = float(np.std(np.exp(1j * ph_gated))) if False else float(
        math.sqrt(max(0.0, -2 * math.log(max(abs(np.mean(np.exp(1j * ph_gated))), 1e-12)))))
    ph4 = np.angle(np.exp(1j * 4 * ph_gated))
    sigma_dp = float(math.sqrt(max(0.0, -2 * math.log(max(abs(np.mean(np.exp(1j * ph4))), 1e-12)))))
    # higher-order phase concentration (how tightly the phase sits on 2/4/8 points)
    cluster_scores = {}
    for k in (2, 4, 8):
        r = float(abs(np.mean(np.exp(1j * k * ph_gated))))
        cluster_scores[f"phase_cluster_{k}"] = r

    # instantaneous frequency features
    phu = np.unwrap(np.angle(x))
    dph = np.diff(phu)
    slope = float(np.polyfit(np.arange(dph.size, dtype=np.float64), dph, 1)[0]) if dph.size > 8 else 0.0
    inst = (dph - slope) * fs / (2 * math.pi)
    if inst.size > 8:
        inst_s = np.convolve(inst, np.ones(5) / 5, mode="same")[4:-4]
    else:
        inst_s = inst
    ratio_ok = (rs is not None and rs > 0)
    sigma_af = float(np.std(inst_s) / (obw_hz / 2.0)) if obw_hz else (
        float(np.std(inst_s) / max(np.mean(np.abs(inst_s)), 1e-9)))
    # bimodality of the IF histogram: FSK has two levels, PSK has a narrow distribution
    if inst_s.size > 64:
        hist, edges = np.histogram(inst_s, bins=32)
        hist = hist / max(hist.sum(), 1)
        nz = hist[hist > 0]
        flatness_if = float(np.std(hist) / max(np.mean(hist), 1e-12))
        # count modes with a simple peak count on a smoothed histogram
        sm = np.convolve(hist, np.ones(3) / 3, mode="same")
        peaks = int(np.sum((sm[1:-1] > sm[:-2]) & (sm[1:-1] >= sm[2:]) & (sm[1:-1] > 0.5 * sm.max())))
    else:
        flatness_if, peaks = 0.0, 0

    # envelope-spectrum tone (AM message tone / ASK keying rate)
    env = mag - np.mean(mag)
    fe, Pe = sigproc.welch(env, fs=fs, nperseg=min(nper, max(128, env.size // 4)),
                           window=make_window("hann", min(nper, max(128, env.size // 4))),
                           return_onesided=True, scaling="density")
    if Pe.size > 8:
        band = (fe > max(16.0, fs / 5000.0)) & (fe < fs / 2)
        if np.any(band):
            tone_peak = float(np.max(Pe[band]))
            tone_floor = float(np.median(Pe[band]))
            envelope_tone_db = 10 * math.log10(max(tone_peak, 1e-30) / max(tone_floor, 1e-30))
            envelope_tone_hz = float(fe[band][int(np.argmax(Pe[band]))])
        else:
            envelope_tone_db, envelope_tone_hz = 0.0, None
    else:
        envelope_tone_db, envelope_tone_hz = 0.0, None

    # carrier (DC) line strength relative to the in-band density
    nper2 = int(min(4096, max(128, n // 4)))
    fx, Px = sigproc.welch(x, fs=fs, nperseg=nper2, window=make_window("hann", nper2),
                           return_onesided=False, scaling="density")
    order = np.argsort(fx)
    fx, Px = fx[order], Px[order]
    i0 = int(np.argmin(np.abs(fx)))
    local = float(np.median(Px[max(0, i0 - 40): i0 + 41])) if Px.size > 80 else float(np.median(Px))
    dc_line_db = float(10 * math.log10(max(float(Px[i0]), 1e-30) / max(local, 1e-30)))

    band_mask = np.ones(Px.size, dtype=bool) if not obw_hz else np.abs(fx) <= obw_hz / 2
    pb = Px[band_mask]
    spectral_flatness = float(np.exp(np.mean(np.log(np.maximum(pb, 1e-30)))) / max(np.mean(pb), 1e-30)) \
        if pb.size > 8 else None

    # Low-pass filtered instantaneous frequency (cut-off at 0.7 x Rs when known, else 0.1 x BW).
    # Filtering removes broadband phase noise, which is what makes the within-symbol constancy
    # metric usable as an FSK discriminator at moderate SNR.
    try:
        cut_hz = 0.7 * rs if (rs and rs > 0) else (0.1 * (obw_hz or fs / 8.0))
        cut = float(np.clip(cut_hz / (fs / 2.0), 1e-4, 0.95))
        sos = sigproc.butter(4, cut, btype="lowpass", output="sos")
        inst_lp = sigproc.sosfilt(sos, inst if inst.size > 32 else np.zeros(64))
    except Exception:
        inst_lp = inst
    if_constancy_lp = None
    if rs and rs > 0 and inst_lp.size > 64:
        sps_f = fs / rs
        if 2 < sps_f < 4096:
            best = 0.0
            for ph_off in np.linspace(0.0, 1.0, 12, endpoint=False):
                n_sym = int(inst_lp.size / sps_f)
                if n_sym < 16:
                    continue
                idx_s = np.floor((np.arange(n_sym) + ph_off) * sps_f).astype(int)
                idx_s = idx_s[idx_s + int(sps_f) < inst_lp.size]
                if idx_s.size < 16:
                    continue
                seg = np.stack([inst_lp[i:i + int(sps_f)] for i in idx_s])
                w = float(np.mean(seg.var(axis=1)))
                bt = float(np.var(seg.mean(axis=1)))
                best = max(best, bt / max(w, 1e-12))
            if_constancy_lp = float(best)
    if_bw_over_obw = None
    if obw_hz and inst_lp.size > 256:
        try:
            nper_if = int(min(8192, max(256, inst_lp.size // 4)))
            fi, Pi = sigproc.welch(inst_lp - np.mean(inst_lp), fs=fs, nperseg=nper_if,
                                   window=make_window("hann", nper_if), scaling="density")
            ci = np.cumsum(Pi) / max(Pi.sum(), 1e-30)
            lo_i = int(np.searchsorted(ci, 0.05))
            hi_i = int(np.searchsorted(ci, 0.95))
            if_bw_over_obw = float((fi[hi_i] - fi[lo_i]) / max(obw_hz, 1e-9))
        except Exception:
            if_bw_over_obw = None
    # Block-wise phase concentration: computing |E[e^{jk*phi}]| on short blocks and taking the
    # median makes the phase features robust to a residual carrier offset (which rotates the
    # phase slowly across the record).
    blk_scores = {2: None, 4: None, 8: None, 16: None}
    if x.size >= 2048:
        nblk = int(min(12, max(3, x.size // 1024)))
        edges = np.linspace(0, x.size, nblk + 1).astype(int)
        acc = {k: [] for k in blk_scores}
        for i in range(nblk):
            seg_x = x[edges[i]:edges[i + 1]]
            if seg_x.size < 64:
                continue
            seg_mag = np.abs(seg_x)
            g2 = seg_mag >= 0.35 * np.max(seg_mag)
            if np.count_nonzero(g2) < 32:
                continue
            p2 = np.unwrap(np.angle(seg_x[g2]))
            # Remove the linear phase ramp of the block (residual carrier offset) before looking
            # for constellation clustering; without this the phase clusters are washed out.
            t2 = np.arange(p2.size, dtype=float)
            try:
                slope = float(np.polyfit(t2, p2, 1)[0])
            except Exception:
                slope = 0.0
            p2 = p2 - slope * t2
            p2 = p2 - np.angle(np.mean(np.exp(1j * p2)))
            for k in blk_scores:
                acc[k].append(float(abs(np.mean(np.exp(1j * k * p2)))))
        for k in blk_scores:
            if acc[k]:
                blk_scores[k] = float(np.median(acc[k]))
    # IF edge sharpness: how abrupt the instantaneous-frequency transitions are,
    # normalised by (deviation x symbol rate) - separates hard-switched 2FSK from Gaussian-smoothed GFSK
    if_edge = None
    if rs and rs > 0 and inst.size > 64:
        try:
            d_if = np.abs(np.diff(inst_lp))
            dev = float(np.percentile(np.abs(inst_lp - np.mean(inst_lp)), 90))
            if_edge = float(np.percentile(d_if, 99) / max(dev * rs / 10.0, 1e-9))
        except Exception:
            if_edge = None
    feats = {
        "sigma_aa": sigma_aa,
        "envelope_ripple_db": float(20 * math.log10(max(np.percentile(mag, 95), 1e-30) /
                                                     max(np.percentile(mag, 5), 1e-30))),
        "gamma_max": gamma_max_db,
        "sigma_ap": sigma_ap,
        "sigma_dp": sigma_dp,
        "sigma_af": sigma_af,
        "mag_kurtosis": float(_sp_kurtosis(p, fisher=True)),
        "spectral_flatness": spectral_flatness,
        "dc_carrier_line_db": dc_line_db,
        "envelope_tone_db": envelope_tone_db,
        **cluster_scores,
        "if_bimodality": float(peaks),
        "if_constancy_lp": if_constancy_lp,
        "if_bw_over_obw": if_bw_over_obw,
        "obw_over_rs": (float(obw_hz / rs) if (obw_hz and rs) else None),
        "phase_cluster_16": float(abs(np.mean(np.exp(1j * 16 * ph_gated)))),
        "phase_cluster_2_blk": blk_scores[2],
        "phase_cluster_4_blk": blk_scores[4],
        "phase_cluster_8_blk": blk_scores[8],
        "if_edge_sharpness": if_edge,
    }
    # Symbol-instant phase clustering: averaging the phase over *every* sample of a pulse-shaped
    # signal dilutes the constellation (the phase moves continuously between symbols), so the
    # clustering is measured on the symbol grid, taking the best of several timing offsets - the
    # standard way this feature is defined in the AMC literature.  These keys are deliberately not
    # part of FEATURE_NAMES, so the trained model's input vector is unchanged.
    sym_cluster = {"phase_cluster_2_sym": None, "phase_cluster_4_sym": None,
                   "phase_cluster_8_sym": None}
    if rs and rs > 0:
        sps_i = int(round(fs / float(rs)))
        if 2 <= sps_i <= 8192:
            # sample at the symbol instants *after* a matched filter: an RRC pulse on its own is
            # not Nyquist, so the samples taken straight from the received waveform carry ISI that
            # destroys the phase clusters (measured: |E[e^{j4phi}]| = 0.04 for QPSK).  Filtering with
            # the matching RRC (the receiver's job anyway) restores them (measured: 1.00).
            zx = x
            try:
                from .demod import as_rrc_matched
                mf = as_rrc_matched(x, fs, float(rs), 0.35, span=6, decimate_to=4)
                if mf.get("ok") and mf.get("samples") is not None:
                    cand = np.asarray(mf["samples"], dtype=np.complex128)
                    if cand.size >= 128:
                        zx = cand
                        sps_i = int(round(fs / float(rs)))
            except Exception:
                zx = x
            best = {2: 0.0, 4: 0.0, 8: 0.0}
            for off in range(min(sps_i, 8)):
                z = zx[off::sps_i]
                if z.size < 64:
                    continue
                az = np.abs(z)
                z = z[az > 0.3 * float(np.mean(az))]
                if z.size < 48:
                    continue
                php = np.angle(z)
                for m in (2, 4, 8):
                    v = float(abs(np.mean(np.exp(1j * m * php))))
                    if v > best[m]:
                        best[m] = v
            sym_cluster = {"phase_cluster_2_sym": best[2], "phase_cluster_4_sym": best[4],
                           "phase_cluster_8_sym": best[8]}
    feats.update(sym_cluster)
    # Symbol-rate-free IF "switching rate": the 90th percentile of |d(IF)| divided by the IF
    # standard deviation.  A frequency-shift keyed signal jumps between levels (larger value) while
    # an analogue FM signal sweeps smoothly (smaller value) - useful when no symbol rate is known.
    try:
        inst_d = np.diff(inst_s) if inst_s.size > 8 else np.zeros(0)
        feats["if_constancy_free"] = (float(np.percentile(np.abs(inst_d), 90) /
                                            max(float(np.std(inst_s)), 1e-9))
                                      if inst_d.size else None)
    except Exception:
        feats["if_constancy_free"] = None
    return {"ok": True, "features": feats, "extras": {
        "envelope_tone_hz": envelope_tone_hz,
        "if_modes": peaks,
        "if_std_hz": float(np.std(inst_s)) if inst_s.size else None,
        "if_mean_abs_hz": float(np.mean(np.abs(inst_s))) if inst_s.size else None,
        "instantaneous_frequency_hz": inst_s[: min(4096, inst_s.size)].tolist() if inst_s.size else [],
    }}


# --------------------------------------------------------------------------- #
#  Rule-based scoring
# --------------------------------------------------------------------------- #
def rule_scores(f: dict, extras: dict, obw_hz: float | None, rs: float | None,
                x: np.ndarray, fs: float) -> dict:
    """Transparent rule-based votes for each modulation family.

    Each rule contributes a documented weight; scores are returned unnormalised so
    the caller can show the reasoning, together with the rule list that fired.
    """
    scores = {m: 0.0 for m in ALL_MODS}
    fired: list[dict] = []

    def vote(mod: str, w: float, why: str):
        scores[mod] = scores.get(mod, 0.0) + w
        fired.append({"modulation": mod, "weight": round(w, 3), "rule": why})

    s_aa = f["sigma_aa"]
    if s_aa < 0.12:
        vote("BPSK", 0.6, f"constant envelope (sigma_aa={s_aa:.3f}): rules out linear QAM/AM")
        vote("QPSK", 0.6, f"constant envelope (sigma_aa={s_aa:.3f})")
        vote("8PSK", 0.6, f"constant envelope (sigma_aa={s_aa:.3f})")
        vote("2FSK", 0.7, f"constant envelope (sigma_aa={s_aa:.3f})")
        vote("GFSK", 0.7, f"constant envelope (sigma_aa={s_aa:.3f})")
        vote("FM", 0.5, f"constant envelope (sigma_aa={s_aa:.3f})")
    elif s_aa < 0.30:
        vote("16QAM", 0.55, f"moderate amplitude variation (sigma_aa={s_aa:.3f}): typical of 16-QAM")
        vote("64QAM", 0.45, f"moderate amplitude variation (sigma_aa={s_aa:.3f})")
        vote("AM", 0.25, f"moderate amplitude variation (sigma_aa={s_aa:.3f})")
        vote("2FSK", 0.2, f"amplitude variation is unusual for constant-envelope FSK (sigma_aa={s_aa:.3f})")
    else:
        vote("AM", 1.0, f"large amplitude variation (sigma_aa={s_aa:.3f}): amplitude or ASK-like modulation")
        vote("64QAM", 0.3, f"large amplitude variation (sigma_aa={s_aa:.3f})")
        for m in ("BPSK", "QPSK", "8PSK", "2FSK", "GFSK", "FM"):
            vote(m, -0.5, f"amplitude variation too large for a constant-envelope {m} (sigma_aa={s_aa:.3f})")

    if extras.get("if_std_hz") and obw_hz:
        if_frac = extras["if_std_hz"] / max(obw_hz / 2.0, 1e-9)
        # The width of the instantaneous-frequency distribution alone is NOT FSK evidence: a
        # band-limited PSK/QAM signal sweeps widely during every phase transition, so a wide IF
        # spread also occurs for 8PSK/16QAM.  The physical discriminator is whether the IF stays
        # *piecewise constant between transitions* (digital FSK) or varies continuously (PSK/QAM),
        # which the constancy metric measures.  A wide spread without piecewise constancy is
        # therefore treated as multi-level phase evidence, not as FSK.
        _const = extras.get("fsk_constancy")
        _piecewise = (_const is None) or (float(_const) > 1.80)
        if if_frac > 0.25 and _piecewise:
            vote("2FSK", 0.8, f"large instantaneous-frequency spread ({if_frac:.2f} of the half-bandwidth)")
            vote("GFSK", 0.8, f"large instantaneous-frequency spread ({if_frac:.2f} of the half-bandwidth)")
            for m in ("BPSK", "QPSK", "8PSK"):
                vote(m, -0.7, f"instantaneous-frequency spread too large for {m}")
            vote("16QAM", -0.3, "instantaneous-frequency spread larger than expected for QAM")
            vote("64QAM", -0.3, "instantaneous-frequency spread larger than expected for QAM")
        elif if_frac > 0.25:
            vote("8PSK", 0.50, f"wide instantaneous-frequency spread ({if_frac:.2f} of the "
                              f"half-bandwidth) without piecewise-constant segments: band-limited "
                              f"multi-level phase transitions, not FSK")
            vote("QPSK", 0.35, f"wide instantaneous-frequency spread ({if_frac:.2f}) from phase "
                               f"transitions (not piecewise constant)")
            vote("2FSK", -0.55, f"instantaneous frequency is not piecewise constant "
                                f"(constancy metric {float(_const):.2f}): not digital FSK")
            vote("GFSK", -0.55, f"instantaneous frequency is not piecewise constant "
                                f"(constancy metric {float(_const):.2f})")
        elif if_frac < 0.10:
            for m in ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM"):
                vote(m, 0.35, f"stable instantaneous frequency ({if_frac:.2f} of the half-bandwidth)")
            vote("2FSK", -0.6, f"instantaneous frequency is stable ({if_frac:.2f}): not FSK")
            vote("GFSK", -0.6, f"instantaneous frequency is stable ({if_frac:.2f}): not FSK")

    # phase-cluster structure: the symbol-instant clusters (measured after matched filtering) are
    # the decisive ones when a symbol rate is known; the sample-wise clusters are only a fallback
    c2s, c4s, c8s = (f.get("phase_cluster_2_sym"), f.get("phase_cluster_4_sym"),
                    f.get("phase_cluster_8_sym"))
    sym_ok = all(v is not None for v in (c2s, c4s, c8s)) and (c2s + c4s + c8s) > 0.15
    if sym_ok:
        if c2s > 0.7 and c4s > 0.7:
            vote("BPSK", 1.6, f"symbol-instant phases lie on a single axis "
                              f"(|E[e^(j2phi)]|={c2s:.2f} and |E[e^(j4phi)]|={c4s:.2f} after matched "
                              f"filtering): BPSK")
            for m in ("QPSK", "8PSK", "16QAM", "64QAM"):
                vote(m, -0.5, f"single-axis symbol phases ({c2s:.2f}, {c4s:.2f}) rule out {m}")
        elif c4s > 0.7 and c2s < 0.45:
            vote("QPSK", 1.6, f"symbol-instant phases are on a 4-point grid "
                              f"(|E[e^(j4phi)]|={c4s:.2f}, |E[e^(j8phi)]|={c8s:.2f}) and show no "
                              f"2-fold symmetry (c2={c2s:.2f}): QPSK")
            for m in ("BPSK", "8PSK", "16QAM", "64QAM"):
                vote(m, -0.5, f"4-point symbol-phase grid (c4={c4s:.2f}) rules out {m}")
        elif c8s > 0.7 and c4s < 0.45:
            vote("8PSK", 1.6, f"symbol-instant phases are on an 8-point grid "
                              f"(|E[e^(j8phi)]|={c8s:.2f}) with no 4-fold symmetry: 8PSK")
            for m in ("BPSK", "QPSK", "16QAM", "64QAM"):
                vote(m, -0.5, f"8-point symbol-phase grid (c8={c8s:.2f}) rules out {m}")
        elif c4s < 0.5 and c8s < 0.5 and c2s < 0.5:
            vote("16QAM", 0.5, f"symbol-instant phases form no PSK grid (c2={c2s:.2f}, "
                               f"c4={c4s:.2f}, c8={c8s:.2f}): QAM-like")
            vote("64QAM", 0.5, f"symbol-instant phases form no PSK grid (c2={c2s:.2f}, "
                               f"c4={c4s:.2f}, c8={c8s:.2f}): QAM-like")
    c2, c4, c8 = f["phase_cluster_2"], f["phase_cluster_4"], f["phase_cluster_8"]
    if c2 > 0.75 and c4 < 0.4:
        vote("BPSK", 0.9, f"phase concentrated on 2 points (|E[e^(j2phi)]|={c2:.2f} vs "
                          f"|E[e^(j4phi)]|={c4:.2f})")
    if c4 > 0.6:
        vote("QPSK", 0.9, f"phase concentrated on 4 points (|E[e^(j4phi)]|={c4:.2f})")
        vote("BPSK", -0.4, f"4-fold phase symmetry present ({c4:.2f}), which BPSK does not produce")
    if c8 > 0.55:
        vote("8PSK", 0.7, f"8-fold phase symmetry present (|E[e^(j8phi)]|={c8:.2f})")
    if c2 < 0.3 and c4 < 0.35 and s_aa < 0.3:
        vote("16QAM", 0.35, f"no dominant PSK phase symmetry (c2={c2:.2f}, c4={c4:.2f}, c8={c8:.2f})")
        vote("64QAM", 0.45, f"no dominant PSK phase symmetry (c2={c2:.2f}, c4={c4:.2f}, c8={c8:.2f})")

    # envelope tone / carrier line (AM family)
    # A dominant envelope tone is decisive amplitude-modulation evidence on its own.  The carrier
    # line that accompanies it is measured *after* band extraction and DC removal, so its absolute
    # level is not a reliable gate (measured on a 70 %-index AM tone: envelope tone +54 dB, carrier
    # line -4.6 dB because the carrier sits at 0 Hz and the preprocessing removes it).  The tone is
    # therefore the primary condition and the carrier line only strengthens it.
    _tone_hz = f.get("envelope_tone_hz")
    _tone_at_rs = bool(rs and _tone_hz and abs(float(_tone_hz) - float(rs)) <= 0.10 * float(rs))
    if f["envelope_tone_db"] > 45 and s_aa > 0.30 and not _tone_at_rs:
        # Measured on the generator's ground truth at 22 dB SNR: the envelope tone of amplitude
        # modulation reaches +54 dB while every pulse-shaped linear modulation tops out at +34 dB
        # (their envelope line sits at the symbol rate, from the pulse shaping - not a message
        # tone).  A 45 dB threshold therefore separates AM from the linear family with >=10 dB of
        # margin, and an AM message tone is never at the symbol rate.
        vote("AM", 2.4, f"envelope-spectrum tone {f['envelope_tone_db']:.1f} dB above the envelope "
                        f"floor (the linear family measures at most ~34 dB, whose line sits at the "
                        f"symbol rate) with amplitude variation sigma_aa={s_aa:.3f}: amplitude "
                        f"modulation")
        vote("2FSK", -0.5, "a dominant envelope tone is inconsistent with constant-envelope FSK")
        vote("GFSK", -0.5, "a dominant envelope tone is inconsistent with constant-envelope GFSK")
        vote("FM", -0.5, "a dominant envelope tone is inconsistent with constant-envelope FM")
    elif f["envelope_tone_db"] > 25 and s_aa > 0.30 and _tone_at_rs:
        fired.append({"modulation": None, "weight": 0.0,
                      "rule": f"envelope tone {f['envelope_tone_db']:.1f} dB sits at the symbol rate "
                              f"({float(_tone_hz):,.0f} Hz): the pulse-shaping line of a linear "
                              f"modulation, not an AM message tone"})
    if f["envelope_tone_db"] > 10 and f["dc_carrier_line_db"] > 8:
        vote("AM", 1.2, f"envelope-spectrum tone {f['envelope_tone_db']:.1f} dB above the envelope floor "
                        f"combined with a carrier line {f['dc_carrier_line_db']:.1f} dB above the "
                        f"surrounding PSD: amplitude modulation")
        vote("FM", -0.4, "an envelope tone is inconsistent with constant-envelope FM")
    elif f["envelope_tone_db"] > 10 and f["dc_carrier_line_db"] <= 8:
        vote("16QAM", 0.2, f"envelope tone present but no strong carrier line "
                           f"({f['dc_carrier_line_db']:.1f} dB)")
    if f["gamma_max"] > 6:
        vote("FM", 0.35, f"nonlinear spectral content in |x|^2 (gamma_max={f['gamma_max']:.1f} dB)")
        vote("2FSK", 0.35, f"nonlinear spectral content in |x|^2 (gamma_max={f['gamma_max']:.1f} dB)")

    # IF constancy: digital FSK has piecewise-constant instantaneous frequency.
    # The threshold is calibrated on the generator's ground truth (measured "separation" of the
    # within-symbol IF distribution): BPSK 0.88, QPSK 1.20, 8PSK 1.28, 16QAM 1.18, 64QAM 1.13 versus
    # 2FSK 3.04 and GFSK 2.45 at 22 dB SNR on the same record length.  The separator is therefore
    # 1.80 with a wide margin on both sides - the previous 1.25 threshold sat inside the linear
    # population and made clean 8PSK records flip between 8PSK and 2FSK for a 7 ppm change in the
    # estimated symbol rate.
    const = extras.get("fsk_constancy")
    if const is not None:
        if const > 1.80:
            vote("2FSK", 0.9, f"instantaneous frequency is piecewise constant within symbol intervals "
                              f"(constancy metric {const:.2f} > 1.80, the measured linear/FSK "
                              f"separator): characteristic of FSK")
            vote("GFSK", 0.9, f"instantaneous frequency is piecewise constant (constancy metric {const:.2f})")
            vote("FM", -0.3, f"analogue FM has a continuously varying instantaneous frequency "
                             f"(constancy metric {const:.2f})")
        elif const < 1.55:
            vote("2FSK", -0.5, f"no piecewise-constant instantaneous frequency (constancy metric "
                               f"{const:.2f}): not digital FSK")
            vote("GFSK", -0.5, f"no piecewise-constant instantaneous frequency (constancy metric {const:.2f})")
    if extras.get("if_std_hz") and obw_hz and const is not None and const < 1.55:
        if_frac2 = extras["if_std_hz"] / max(obw_hz / 2.0, 1e-9)
        if if_frac2 > 0.30 and s_aa < 0.15:
            vote("FM", 0.85, f"continuously varying instantaneous frequency with a large deviation "
                             f"({if_frac2:.2f} of the half-bandwidth) and a constant envelope")
        elif if_frac2 < 0.20:
            vote("FM", -0.3, f"instantaneous-frequency deviation too small for analogue FM "
                             f"({if_frac2:.2f} of the half-bandwidth)")

    # A single-tone FM test signal *is* a two-level frequency-shift waveform: with a pure tone
    # message the instantaneous-frequency histogram is bimodal and no waveform measurement can
    # separate it from 2FSK.  Say so instead of hiding it.
    if (s_aa < 0.18 and extras.get("if_std_hz") and obw_hz and
            (extras["if_std_hz"] / max(obw_hz / 2.0, 1e-9)) > 0.25 and
            (extras.get("fsk_constancy") is None or float(extras["fsk_constancy"]) <= 1.80) and
            f["envelope_tone_db"] < 15):
        fired.append({"modulation": None, "weight": 0.0,
                      "rule": "constant envelope with a wide instantaneous-frequency excursion: a "
                              "single-tone analogue FM test signal is waveform-identical to 2FSK, "
                              "so FM is reported as an alternative rather than claimed as the "
                              "primary (zero-weight evidence note)"})

    # spectral flatness: OFDM/noise-like
    if f.get("spectral_flatness") and f["spectral_flatness"] > 0.75:
        fired.append({"modulation": None, "weight": 0.0,
                      "rule": f"in-band spectrum is nearly flat (Wiener entropy "
                              f"{f['spectral_flatness']:.2f}): OFDM/noise-like emissions are outside the "
                              f"classifier's supported set"})
    return {"scores": scores, "rules_fired": fired}


# --------------------------------------------------------------------------- #
#  Constellation evidence via actual demodulation
# --------------------------------------------------------------------------- #
def constellation_evidence(x: np.ndarray, fs: float, rs: float | None, mods: list[str],
                           rolloff: float = 0.35, max_symbols: int = 8000) -> list[dict]:
    """Demodulate each candidate modulation and report the measured EVM."""
    out = []
    xs = x[: min(x.size, 4 * max_symbols)] if rs else x
    for m in mods:
        if m in LINEAR:
            if not rs:
                out.append({"modulation": m, "ok": False, "error": "symbol rate required"})
                continue
            r = dm.demodulate_linear(xs, fs, m, rs, rolloff)
        elif m == "2FSK":
            if not rs:
                out.append({"modulation": m, "ok": False, "error": "symbol rate required"})
                continue
            r = dm.demodulate_fsk(xs, fs, rs)
        else:
            continue
        if not r.get("ok"):
            out.append({"modulation": m, "ok": False, "error": r.get("error")})
            continue
        q = r.get("quality", {})
        if m in LINEAR:
            out.append({"modulation": m, "ok": True, "evm_rms": q.get("evm_rms"),
                        "evm_db": q.get("evm_db"), "cluster_coverage": q.get("cluster_coverage"),
                        "n_symbols": r.get("n_symbols")})
        else:
            ts = dm.fsk_timing_score(xs, fs, rs)
            sep = q.get("separation_ratio")
            levels = r.get("symbols")
            fit = None
            if levels is not None and levels.size > 16:
                lv = np.asarray(levels).real
                d = np.abs(lv - np.mean(lv))
                g1, g2 = lv[d <= np.median(d)], lv[d > np.median(d)]
                if g1.size > 4 and g2.size > 4:
                    between = float(np.var(np.concatenate([np.full(g1.size, g1.mean()),
                                                           np.full(g2.size, g2.mean())])))
                    fit = between / max(float(np.var(lv)), 1e-12)
            out.append({"modulation": m, "ok": True, "evm_rms": None,
                        "separation_ratio": sep, "if_constancy": ts.get("separation"),
                        "two_level_fit": fit, "n_symbols": r.get("n_symbols")})
    return out


# --------------------------------------------------------------------------- #
#  Main classifier
# --------------------------------------------------------------------------- #
def _load_model() -> tuple[Any, str | None]:
    try:
        import joblib
        if os.path.exists(MODEL_PATH):
            b = joblib.load(MODEL_PATH)
            return b, None
        return None, f"no trained model found at {os.path.basename(MODEL_PATH)}"
    except Exception as exc:
        return None, f"model load failed: {exc}"


def classify(x: np.ndarray, fs: float, rs: float | None = None, obw_hz: float | None = None,
             snr_db: float | None = None, rolloff: float = 0.35, use_ml: bool = True,
             run_demod_evidence: bool = True, is_real: bool = False,
             carrier_hz: float | None = None) -> dict:
    """Full hybrid classification with evidence, alternatives and limitations.

    Real-valued (passband) captures are converted to their analytic signal with a
    Hilbert transform first, and the residual carrier is removed using the supplied
    (or internally estimated) spectral centroid, so the phase-domain features are
    measured against a stationary constellation rather than a rotating one.
    """
    x = np.asarray(x)
    if x.ndim != 1 or x.size < 512:
        return {"ok": False, "error": "at least 512 samples are required for classification",
                "candidates": [], "params": []}
    notes: list[str] = []
    was_real = not np.iscomplexobj(x)
    if was_real:
        from .preprocess import analytic_signal
        x = analytic_signal(np.asarray(x, dtype=np.float64))
        notes.append("real-valued input converted to its analytic signal (Hilbert transform) so that "
                     "complex-domain features are well defined")
    x = np.asarray(x, dtype=np.complex128)
    # Remove the residual carrier: phase-domain features require a stationary constellation.
    # The offset is measured from the M-th power spectral line whenever that line is strong
    # enough, because the estimator is exact to a few Hz; a supplied (or internally computed)
    # spectral centroid is used only as an alias anchor and as a fallback.  Measured difference on
    # a 20 dB QPSK record: centroid 433 Hz off the true carrier (which diluted the phase clusters
    # and produced an 8PSK verdict), M-th power line 0.1 Hz off.
    from . import demod as _dm
    hint = None if carrier_hz is None else float(carrier_hz)
    carrier_used, carrier_method = None, None
    try:
        est = _dm.estimate_carrier_mth_power(x, fs, 4)
        if est.get("ok") and float(est.get("line_strength_db") or 0.0) >= 6.0:
            f_m = float(est["frequency_offset_hz"])
            if hint is not None:
                alias = _dm.resolve_carrier_alias(f_m, fs, 4, hint)
                if alias.get("ok"):
                    f_m = float(alias["frequency_offset_hz"])
                carrier_method = (f"residual carrier measured from the 4th-power spectral line "
                                  f"({est['line_strength_db']:.1f} dB above the local floor), alias "
                                  f"resolved against the {hint:.1f} Hz spectral-centroid hint")
            else:
                carrier_method = (f"residual carrier measured from the 4th-power spectral line "
                                  f"({est['line_strength_db']:.1f} dB above the local floor)")
            carrier_used = f_m
    except Exception:
        carrier_used = None
    if carrier_used is None:
        if hint is None:
            try:
                from . import spectrum as _sp
                a = _sp.analyse_spectrum(x, fs, nperseg=int(min(4096, max(256, x.size // 8))))
                hint = float(a.get("carrier_hz", 0.0)) if a.get("ok") else 0.0
                carrier_method = (f"no usable 4th-power line: residual carrier taken from the "
                                  f"spectral centroid ({hint:.1f} Hz), estimated internally")
            except Exception:
                hint, carrier_method = 0.0, "carrier removal skipped (no estimator available)"
        else:
            carrier_method = (f"no usable 4th-power line: residual carrier taken from the supplied "
                              f"spectral centroid ({hint:.1f} Hz)")
        carrier_used = hint
    carrier_hz = float(carrier_used or 0.0)
    notes.append(carrier_method or "carrier removal not applied")
    if abs(carrier_hz) > 1e-9:
        t = np.arange(x.size, dtype=np.float64) / fs
        x = x * np.exp(-2j * np.pi * float(carrier_hz) * t)
    fe = extract_features(x, fs, rs, obw_hz)
    if not fe.get("ok"):
        return {"ok": False, "error": fe.get("error"), "candidates": [], "params": []}
    f, extras = fe["features"], fe["extras"]
    if rs and np.iscomplexobj(x):
        try:
            ts = dm.fsk_timing_score(x, fs, rs)
            extras["fsk_constancy"] = ts.get("separation") if ts.get("ok") else None
        except Exception:
            extras["fsk_constancy"] = None
    rscores = rule_scores(f, extras, obw_hz, rs, x, fs)
    scores = rscores["scores"]
    if was_real:
        # an analogue passband capture is not a baseband constellation recording: keep the
        # analogue and frequency-shift families, down-weight the suppressed-carrier constellations
        for m in ("16QAM", "64QAM", "8PSK"):
            scores[m] = scores.get(m, 0.0) - 0.6
        notes.append("input was real-valued: analogue (AM/FM) and frequency-shift hypotheses are "
                     "favoured over suppressed-carrier constellations")

    demod_ev: list[dict] = []
    evm_scores: dict[str, float] = {}
    if run_demod_evidence:
        demod_ev = constellation_evidence(x, fs, rs, [m for m in LINEAR] + ["2FSK"], rolloff,
                                          max_symbols=6000)
        ok = [d for d in demod_ev if d.get("ok") and d.get("evm_db") is not None]
        if ok:
            best_db = min(d["evm_db"] for d in ok)      # lowest EVM = best fit (EVM in dB is negative)
            for d in ok:
                penalty = {2: 0.0, 4: 0.6, 8: 1.4, 16: 2.6, 64: 4.2}[
                    {"BPSK": 2, "QPSK": 4, "8PSK": 8, "16QAM": 16, "64QAM": 64}[d["modulation"]]]
                # score falls off with the EVM gap to the best-fitting candidate, plus an explicit
                # complexity penalty so a higher-order constellation must fit *demonstrably* better
                evm_scores[d["modulation"]] = math.exp(-max(0.0, d["evm_db"] - best_db) / 4.0) \
                    * math.exp(-penalty / 2.0)
        # FSK evidence is scored on the measured within-symbol instantaneous-frequency constancy
        # (a true FSK yields piecewise-constant IF segments; PSK/QAM do not) and on how well the
        # recovered symbol levels separate into two groups.  The mapping below is scaled so that
        # observed values for true FSK (constancy ~1.2-1.8) land in the same range as the linear
        # EVM scores, and values typical of PSK/QAM (~0.5-1.0) approach zero.
        for d in demod_ev:
            if not d.get("ok"):
                continue
            const = d.get("if_constancy")
            fit = d.get("two_level_fit")
            if const is None:
                continue
            fsk_score = clamp01((const - 0.95) / 1.03) * (0.55 + 0.45 * clamp01((fit or 0.5) / 0.85))
            evm_scores["2FSK"] = fsk_score
            evm_scores["GFSK"] = fsk_score * 0.98

    ml_probs: dict[str, float] = {}
    ml_note = None
    if use_ml:
        bundle, err = _load_model()
        if bundle is None:
            ml_note = (f"machine-learning tie-breaker unavailable ({err}); classification is DSP-only. "
                       f"Run `python ml/train_classifier.py` to train the classical model on synthetic data.")
        else:
            try:
                vec = np.array([[f.get(k, 0.0) or 0.0 for k in FEATURE_NAMES]], dtype=np.float64)
                clf = bundle["model"]
                proba = clf.predict_proba(vec)[0]
                for cls, p in zip(clf.classes_, proba):
                    ml_probs[str(cls)] = float(p)
                ml_note = (f"classical ML model '{bundle.get('name', 'rf')}' trained on "
                           f"{bundle.get('n_train', '?')} synthetic vectors "
                           f"({bundle.get('accuracy', float('nan')):.1%} held-out accuracy)")
            except Exception as exc:
                ml_note = f"ML inference failed ({exc}); DSP-only classification used"

    # ---- fuse: normalised rule score, EVM score, ML probability
    mods = list(ALL_MODS)
    rnorm = {}
    vals = np.array([scores.get(m, 0.0) for m in mods], dtype=np.float64)
    if vals.size:
        shifted = vals - vals.min()
        exp = np.exp(shifted / max(0.35, np.std(shifted) if np.std(shifted) > 0 else 1.0))
        rnorm = {m: float(e / exp.sum()) for m, e in zip(mods, exp)}
    if ml_probs:
        w_rules = 0.35
        w_evm = 0.35 if evm_scores else 0.65
        # the ML weight is only what is actually available to add: an unavailable model must not
        # silently dilute the DSP weights (it made every fused score smaller for no reason)
        w_ml = max(0.0, 1.0 - w_rules - w_evm)
    else:
        w_rules = 0.50
        w_evm = 0.50 if evm_scores else 0.0
        w_ml = 0.0
    fused = {}
    for m in mods:
        fused[m] = (w_rules * rnorm.get(m, 0.0)
                    + w_evm * evm_scores.get(m, 0.0)
                    + w_ml * ml_probs.get(m, 0.0))
    tot = sum(fused.values())
    if tot <= 0:
        return {"ok": False, "error": "no modulation hypothesis could be scored",
                "candidates": [], "params": [], "features": f, "notes": notes}
    probs = {m: v / tot for m, v in fused.items()}
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])

    # ---- confidence: softmax margin + SNR gating + ambiguity penalty
    top_mod, top_p = ranked[0]
    second_p = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_p - second_p
    conf = clamp01(0.30 + 0.55 * margin / max(top_p, 1e-6) + 0.25 * (top_p - 1.0 / len(mods)) / 0.6)
    if snr_db is not None:
        if snr_db < 5:
            conf *= 0.45
        elif snr_db < 10:
            conf *= 0.75
    if f.get("spectral_flatness") and f["spectral_flatness"] > 0.75:
        conf *= 0.6
    conf = min(conf, 0.93)          # never present a blind classification as certain
    ambiguous = margin < 0.15

    evidence_common = [
        f"envelope variation sigma_aa = {f['sigma_aa']:.3f} ({'constant envelope' if f['sigma_aa'] < 0.15 else 'amplitude variation present'})",
        f"2-fold / 4-fold / 8-fold phase concentration = {f['phase_cluster_2']:.2f} / "
        f"{f['phase_cluster_4']:.2f} / {f['phase_cluster_8']:.2f}",
        f"in-band spectral flatness (Wiener entropy) = "
        f"{f['spectral_flatness']:.2f}" if f.get("spectral_flatness") is not None else "spectral flatness unavailable",
        f"|x|^2 spectrum peak-to-mean gamma_max = {f['gamma_max']:.1f} dB",
        (f"instantaneous-frequency std = {extras['if_std_hz']:.1f} Hz"
         if extras.get("if_std_hz") is not None else "instantaneous frequency unavailable"),
    ]
    if rs:
        evidence_common.append(f"all hypotheses demodulated at the estimated symbol rate {rs:g} sym/s")
    if obw_hz:
        evidence_common.append(f"occupied bandwidth {obw_hz:g} Hz used as the spectral reference")

    candidates = []
    for mod, p in ranked[:5]:
        ev = list(evidence_common)
        for r in rscores["rules_fired"]:
            if r["modulation"] == mod:
                ev.append(f"rule ({r['weight']:+.2f}): {r['rule']}")
        for d in demod_ev:
            if d["modulation"] == mod and d.get("ok"):
                if d.get("evm_db") is not None:
                    ev.append(f"measured constellation EVM after blind demodulation: "
                              f"{100*d['evm_rms']:.2f}% ({d['evm_db']:.1f} dB)")
                if d.get("separation_ratio") is not None:
                    ev.append(f"FSK integrate-and-dump level separation ratio "
                              f"{d['separation_ratio']:.2f}")
        lim = []
        if ambiguous:
            lim.append("the top two hypotheses are within 15 percentage points of each other: "
                       "treat the ordering as provisional")
        if snr_db is not None and snr_db < 10:
            lim.append(f"SNR {snr_db:.1f} dB is low: feature estimators are biased in this regime")
        if mod in ("16QAM", "64QAM") and snr_db is not None and snr_db < 15:
            lim.append("higher-order QAM classification is unreliable below ~15 dB in-band SNR")
        if mod in ("8PSK", "16QAM", "64QAM") and snr_db is not None and snr_db < 12:
            lim.append("blind carrier recovery for this order degrades below ~12 dB")
        if top_mod == mod and ml_probs:
            lim.append("the ML tie-breaker cannot be validated on real data: it is trained on synthetic signals")
        candidates.append({
            "modulation": mod,
            "probability": round(float(p), 4),
            "confidence_pct": round(100 * float(p), 1),
            "evidence": ev[:10],
            "limitations": lim,
            "rule_score": round(float(scores.get(mod, 0.0)), 3),
            "evm_score": round(float(evm_scores.get(mod, 0.0)), 4),
            "ml_probability": round(float(ml_probs.get(mod, 0.0)), 4) if ml_probs else None,
        })

    params = [
        param("Modulation (primary hypothesis)", top_mod, None, conf,
              f"hybrid classification: DSP rules (weight {w_rules:.2f}), measured constellation EVM "
              f"evidence (weight {w_evm:.2f}), ML tie-breaker (weight {w_ml:.2f})",
              evidence=candidates[0]["evidence"][:6],
              limitations=candidates[0]["limitations"],
              status=STATUS_OK if conf > 0.5 and not ambiguous else STATUS_LOW),
        param("Modulation - alternatives",
              [{"modulation": c["modulation"], "probability": c["probability"]} for c in candidates[1:4]],
              None, None, "same hybrid classifier; all hypotheses retained for multi-hypothesis analysis",
              note="alternatives are not discarded: the hypothesis engine demodulates each of them"),
        param("Classifier inputs",
              {"snr_db": snr_db, "symbol_rate_hz": rs, "occupied_bandwidth_hz": obw_hz},
              None, None, "context supplied to the classifier"),
    ]
    return {
        "ok": True, "primary": top_mod, "confidence": round(float(conf), 4), "notes": notes,
        "carrier_removed_hz": float(carrier_hz or 0.0),
        "candidates": candidates, "features": f, "feature_extras": extras,
        "rules_fired": rscores["rules_fired"], "demodulation_evidence": demod_ev,
        "ml": {"available": bool(ml_probs), "probabilities": ml_probs, "note": ml_note,
               "weight": w_ml},
        "weights": {"rules": w_rules, "evm": w_evm, "ml": w_ml},
        "ambiguous": bool(ambiguous), "params": params,
        "method": "hybrid classifier: transparent DSP feature rules + measured constellation EVM from "
                  "actual blind demodulation of every candidate + optional classical ML tie-breaker",
        "limitations": [
            "supported set: BPSK, QPSK, 8PSK, 16QAM, 64QAM, 2FSK, GFSK, AM, FM "
            "(plus FM broadcast stereo MPX). OFDM, spread spectrum, APSK and continuous-phase variants "
            "outside GFSK are not classified",
            "classification is blind: no training on the operator's signals and no protocol knowledge",
            "the confidence reflects the internal score margin, not a calibrated error rate",
        ],
    }


# --------------------------------------------------------------------------- #
#  Constellation and eye-diagram views
# --------------------------------------------------------------------------- #
def constellation_view(x: np.ndarray, fs: float, mod: str, rs: float, rolloff: float = 0.35,
                       view: str = "symbols", max_points: int = 3000) -> dict:
    """I/Q scatter for the UI: raw | matched-filtered | symbol-sampled."""
    x = np.asarray(x, dtype=np.complex128)
    if mod not in LINEAR:
        return {"ok": False, "error": f"constellation analysis is defined for linear modulations, not {mod}"}
    mf = dm.as_rrc_matched(x, fs, rs, rolloff)
    if not mf.get("ok"):
        return {"ok": False, "error": mf.get("error")}
    stages = []
    if view == "raw":
        pts = x - np.mean(x)
        note = "raw complex samples (DC removed)"
    elif view == "filtered":
        pts = mf["samples"] - np.mean(mf["samples"])
        note = mf["method"]
    else:
        bps = dm.blind_phase_search(mf["samples"], mod, 32, 3)
        sym = mf["samples"]
        if bps.get("ok"):
            sym = sym * np.exp(1j * bps["best_phase_rad"])
            stages.append({"stage": "blind phase search", "status": STATUS_OK,
                           "detail": f"{math.degrees(bps['best_phase_rad']):.2f} deg"})
        smp = dm.sample_symbols(sym, mf["sps_in"], 0.0, delay_samples=mf["delay_samples"])
        if not smp.get("ok"):
            return {"ok": False, "error": smp.get("error")}
        pts = smp["symbols"]
        note = "symbol-sampled constellation after blind phase search; symbol timing from the DFT phase " \
               "of the |x|^2 line (Oerder-Meyr criterion)"
    from .utils import downsample_complex as _ds
    pts = _ds(pts, max_points) if view != "symbols" else pts[:max_points]
    q = dm.constellation_quality(pts, mod)
    const = synth.constellation(mod)
    return {
        "ok": True, "view": view, "i": pts.real.tolist(), "q": pts.imag.tolist(),
        "n_points": int(pts.size), "reference_i": const.real.tolist(), "reference_q": const.imag.tolist(),
        "quality": q, "note": note, "stages": stages,
        "downsampled": bool(pts.size >= max_points),
    }


def eye_diagram(x: np.ndarray, fs: float, rs: float | None = None, rolloff: float = 0.35,
                span_symbols: int = 2, n_traces: int = 120, max_points_per_trace: int = 200) -> dict:
    """Eye diagram from the matched-filtered signal plus timing-quality metrics."""
    x = np.asarray(x, dtype=np.complex128)
    if not rs:
        return {"ok": False, "error": "symbol rate required - run symbol-rate estimation first",
                "estimation_mode": True}
    mf = dm.as_rrc_matched(x, fs, rs, rolloff)
    if not mf.get("ok"):
        return {"ok": False, "error": mf.get("error")}
    y = mf["samples"]
    sps_float = mf["sps_in"]
    per = max(4, int(round(span_symbols * sps_float)))
    tm = dm.oerder_meyr_timing(x, fs, rs)
    phase = tm.get("timing_phase_frac", 0.0) if tm.get("ok") else 0.0
    start = int(mf["delay_samples"] + phase * sps_float)
    n_tr = int(min(n_traces, max(1, (y.size - start - per) // max(1, int(sps_float)))))
    if n_tr < 4:
        return {"ok": False, "error": "not enough symbols for an eye diagram"}
    traces = np.empty((n_tr, min(per, max_points_per_trace)), dtype=np.complex128)
    decim = max(1, int(math.ceil(per / max_points_per_trace)))
    for k in range(n_tr):
        s = start + int(k * sps_float)
        seg = y[s: s + per][::decim][:traces.shape[1]]
        traces[k, :seg.size] = seg
    # timing quality metrics: eye opening at the sampling instant vs the worst instant
    idx = int(round(phase * sps_float / decim))
    idx = max(0, min(idx, traces.shape[1] - 1))
    centre_power = float(np.mean(np.abs(traces[:, idx]) ** 2))
    # worst-case across the whole second symbol interval as a reference for the eye opening
    lo = max(0, idx - traces.shape[1] // 4)
    hi = min(traces.shape[1], idx + traces.shape[1] // 4)
    spread = float(np.mean(np.std(np.abs(traces[:, lo:hi]), axis=0)))
    mean_amp = float(np.mean(np.abs(traces[:, idx])))
    eye_opening = float(mean_amp / max(spread, 1e-9))
    # timing uncertainty from the |x|^2 spectral line width (jitter proxy)
    return {
        "ok": True, "traces_i": traces.real.tolist(), "traces_q": traces.imag.tolist(),
        "n_traces": int(n_tr), "samples_per_trace": int(traces.shape[1]),
        "symbols_per_trace": span_symbols, "time_axis_symbols": (np.arange(traces.shape[1]) * decim / sps_float).tolist(),
        "sampling_instant_index": idx,
        "timing_phase_frac": float(phase),
        "metrics": {
            "eye_opening_ratio": round(eye_opening, 3),
            "mean_symbol_amplitude": mean_amp,
            "trace_spread": spread,
            "timing_line_db": tm.get("timing_line_db") if tm.get("ok") else None,
            "eye_quality": ("open" if eye_opening > 3 else "marginal" if eye_opening > 2 else "closed"),
        },
        "method": "matched-filtered signal sliced at the recovered symbol period; eye opening compares the "
                  "mean symbol amplitude with the trace spread inside +/-quarter symbol",
        "limitations": ["a closed eye can also come from a wrong symbol-rate hypothesis - check the "
                        "symbol-rate alternatives"] if eye_opening < 2 else [],
    }
