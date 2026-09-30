"""Automatic interleaving detection - hypothesis testing with measured statistics.

Interleaving cannot be "seen" directly in a bitstream: it is a permutation.  Two independent,
measurable tests are used instead, and the platform reports which one produced the evidence:

**1. Error-clustering test (structure of the soft decisions).**
Low-reliability symbols (small |LLR|) cluster in time when the channel has memory and no
interleaver is present, and they are spread out when an interleaver disperses them.  A dispersion
test (variance of the unreliable-symbol count across windows versus the Poisson/binomial
expectation) gives a z-score against the memoryless null.

**2. De-interleaving decode test (the decisive one).**
If a forward-error-correction decoder can be run, de-interleaving with the *right* geometry makes
the code constraint suddenly visible: the Viterbi distance to the nearest codeword drops.  Every
candidate geometry from :data:`dsp.synth.INTERLEAVER_CANDIDATES` is applied to the soft values and
the change in the decoding evidence is measured.  A geometry only scores if it *improves* an
already-measurable decode; the platform never infers an interleaver from a decode that was not
working to begin with.

When neither test has power (for example a memoryless AWGN channel with no detected FEC), the
answer is "cannot be determined for this record" - with the reason - rather than a guess.
"""

from __future__ import annotations

import time

import numpy as np

from . import fec as fec_mod
from .synth import INTERLEAVER_CANDIDATES, STANDARD_CONV_CODES, deinterleave
from .utils import clamp01, param, unavailable

__all__ = ["unreliable_symbol_statistics", "clustering_test", "candidate_geometries",
           "test_deinterleaving", "analyse_interleaving", "interleaving_param",
           "apply_deinterleave"]


def unreliable_symbol_statistics(soft: np.ndarray, quantile: float = 0.2) -> dict:
    """Run lengths and gap statistics of the unreliable (|soft| below the ``quantile``) symbols."""
    s = np.abs(np.asarray(soft, dtype=np.float64).ravel())
    n = s.size
    if n < 64:
        return {"ok": False, "message": "not enough soft decisions"}
    thr = float(np.quantile(s, quantile))
    bad = s <= thr
    p = float(np.mean(bad))
    runs, gaps = [], []
    i = 0
    while i < n:
        if bad[i]:
            j = i
            while j < n and bad[j]:
                j += 1
            runs.append(j - i)
            i = j
        else:
            i += 1
    idx = np.flatnonzero(bad)
    if idx.size > 1:
        gaps = np.diff(idx)
    return {"ok": True, "threshold": thr, "p_unreliable": p, "n_unreliable": int(np.count_nonzero(bad)),
            "runs": runs, "mean_run": float(np.mean(runs)) if runs else 0.0,
            "max_run": int(max(runs)) if runs else 0, "n_runs": len(runs),
            "mean_gap": float(np.mean(gaps)) if len(gaps) else 0.0,
            "soft": s, "bad_mask": bad}


def clustering_test(soft: np.ndarray, n_windows: int = 20, quantile: float = 0.2) -> dict:
    """Dispersion test on the unreliable-symbol counts per time window.

    Under the memoryless null the counts are binomial, so ``var/mean ~ 1``.  Bursty error
    clustering inflates it; the z-score is (observed dispersion - 1) / standard error.
    """
    st = unreliable_symbol_statistics(soft, quantile=quantile)
    if not st.get("ok"):
        return st
    s = st["soft"]
    n = s.size
    nw = int(max(4, min(int(n_windows), n // 32)))
    edges = np.linspace(0, n, nw + 1).astype(int)
    counts = np.array([np.count_nonzero(st["bad_mask"][edges[i]:edges[i + 1]]) for i in range(nw)],
                      dtype=float)
    p = st["p_unreliable"]
    mean = float(np.mean(counts))
    var = float(np.var(counts, ddof=1)) if nw > 1 else 0.0
    disp = var / mean if mean > 0 else 1.0
    # standard error of the dispersion under the null (chi2 with nw-1 dof)
    se = float(np.sqrt(2.0 / max(nw - 1, 1)))
    z = (disp - 1.0) / se if se > 0 else 0.0
    return {"ok": True, "dispersion_index": float(disp), "z_score": float(z),
            "counts_per_window": counts.tolist(), "window_size": int(edges[1] - edges[0]),
            "n_windows": nw, "p_unreliable": p, "mean_run": st["mean_run"], "max_run": st["max_run"],
            "n_runs": st["n_runs"],
            "interpretation": ("unreliable symbols are clustered in time (channel memory visible, "
                               "no dispersion effect)" if z > 3 else
                               ("unreliable symbols are spread as a memoryless channel would give "
                                "(dispersion consistent with an interleaver or pure AWGN)"
                                if z < 2 else "clustering is between the two regimes"))}


def candidate_geometries(include_random: bool = True, max_candidates: int = 11) -> list[dict]:
    out = []
    for kind, kw in INTERLEAVER_CANDIDATES:
        if kind == "none":
            continue
        if not include_random and kind == "pseudo-random":
            continue
        label = kind + ("" if not kw else " " + ", ".join(f"{k}={v}" for k, v in kw.items()))
        out.append({"kind": kind, "params": dict(kw), "label": label})
    return out[:max_candidates]


def test_deinterleaving(soft: np.ndarray, bits: np.ndarray | None = None,
                        codes: list | None = None, geometries: list[dict] | None = None,
                        max_bits: int = 8000, max_codes: int = 3,
                        align_offsets: tuple = (), control_permutations: int = 3,
                        n_probe_bits: int = 128, max_probe_offsets: int = 2,
                        stage_bits: int = 2200, final_geometries: int = 2,
                        time_budget_s: float | None = 25.0, seed: int = 20260926) -> dict:
    """Measure how each candidate de-interleaver changes the convolutional decode evidence.

    A wall-clock ``time_budget_s`` bounds the search on small machines: when it expires the
    remaining geometries are skipped and the result says so (``truncated``/``geometries_tested``),
    so a partial search is always distinguishable from a complete one.

    Two-stage measurement: every candidate geometry is ranked on a shorter prefix
    (``stage_bits``, one Viterbi decode per probe offset) and only the best ``final_geometries``
    are re-measured on the full ``max_bits`` stream.  The separation between a correct and an
    incorrect de-interleaver is a factor of ~50 in decoder distance, so the ranking is unchanged
    while the cost drops by roughly 4x - which matters because this test is the most expensive
    stage of the pipeline.  Both lengths are reported with every candidate.
    """
    soft = np.asarray(soft, dtype=np.float64).ravel()
    if bits is None:
        bits = (soft < 0).astype(np.int8)
    bits = np.asarray(bits, dtype=np.int8).ravel()
    n_use = int(min(bits.size, max_bits))
    if n_use < 256:
        return {"ok": False, "message": "not enough bits to test de-interleaving"}
    codes = codes or list(STANDARD_CONV_CODES)[:max_codes]
    geoms = geometries or candidate_geometries()
    geom_by_label = {g["label"]: g for g in geoms}
    stage_use = int(min(n_use, max(1024, stage_bits)))
    # baseline: the stream as received (no interleaver), at both measurement lengths
    base = _best_decode_evidence(bits[:n_use], soft[:n_use], codes)
    base_stage = (_best_decode_evidence(bits[:stage_use], soft[:stage_use], codes)
                  if stage_use < n_use else base)
    # The receiver's bit alignment relative to the transmitter's interleaver blocks is unknown (the
    # demodulator starts at whatever symbol it locked onto).  A block interleaver de-interleaved with
    # the wrong offset restores nothing, so each geometry is tried at a few small offsets - measured
    # on a coded + block-interleaved record: offset 2 improved the decode by 10 % while offset 0 did
    # not change it at all.
    # The demodulated bit stream starts at the first symbol the receiver locked onto, so it can be
    # offset from the transmitter's bit stream by more than a symbol (measured: 18 bits on a 2500
    # symbol record).  The generator's interleavers act on the whole record, so a one-bit misalignment
    # destroys the permutation; the offset search therefore covers small values and the block-size
    # scale.
    # The receiver's start alignment relative to the transmitter's interleaver blocks is unknown and
    # is measured in *bits*: an error of a single bit destroys a block permutation.  The search
    # therefore covers every offset up to the largest plausible start-up loss (measured: 18 bits on a
    # 1500-symbol record) plus the block scales of the catalogue.
    offsets = sorted({int(o) for o in align_offsets} | set(range(0, 64)) |
                     {64, 96, 128, 192, 256})
    def _aligned(bits_arr: np.ndarray, g: dict, off: int) -> np.ndarray:
        """De-interleave after aligning the receiver's stream to the transmitter's blocks.

        ``off`` is the number of bits the receiver lost at the start (its first bit is the
        transmitter's bit ``off``).  The stream is padded with that many placeholders at the front,
        de-interleaved as a whole, and *not* cropped: the padded array is the transmitter's stream
        with unknown values in the first ``off`` positions, so de-interleaving it recovers the coded
        stream with only those few bits (and the bits they scatter onto) wrong.  Cropping the output
        shifted the recovered stream and destroyed the alignment.
        """
        if off <= 0:
            return bits_arr[_perm_for(bits_arr.size, g)]
        padded = np.concatenate([np.zeros(off, dtype=np.int8), bits_arr])
        return padded[_perm_for(bits_arr.size + off, g)]

    results = []
    # The offset probe is deliberately short and tests *every* code in the list: measured cost of one
    # probe decode is ~27 ms at 4096 bits, ~7 ms at 1024 bits and ~4 ms at 512 bits, and the probe only
    # has to *rank* offsets (a correct de-interleaver is separated from a wrong one by a factor of ~50
    # in decoder distance per bit, so a few hundred bits are already decisive).  Testing all the codes
    # matters: when the probe used a single, arbitrary code its distance stayed at the random-stream
    # value for *every* offset, so the ranking became noise (measured on a block-16 record: the probe
    # ranked offset 256 first and the true alignment 16 was never tried).  The top
    # ``max_probe_offsets`` offsets are then confirmed on the full ``stage_bits`` prefix below.
    probe_len = int(min(n_use, max(512, 4 * n_probe_bits)))
    t_start = time.time()
    truncated = False
    tested = 0
    for g in geoms:
        if time_budget_s and (time.time() - t_start) > float(time_budget_s):
            truncated = True
            break
        tested += 1
        # Stage 1: find the alignment offsets with a cheap probe decode (a fraction of the record),
        # then Stage 2: run the full decode only for the best few offsets.  A brute-force full decode
        # over every offset and geometry would cost minutes; the probe costs milliseconds and the
        # separation between "correct alignment" and "wrong alignment" is a factor of ~50 in decoder
        # distance, so nothing is lost.
        scored = []
        probe = bits[:probe_len]
        for off in offsets:
            d_bits = _aligned(probe, g, off)
            ev = _best_decode_evidence(d_bits[:probe_len - off], None, codes)
            if ev.get("ok"):
                scored.append((ev["distance_per_bit"], int(off)))
        if not scored:
            continue
        scored.sort()
        best_for_geometry = None
        for _, off in scored[:max_probe_offsets]:
            d_bits = _aligned(bits[:stage_use], g, off)
            ev = _best_decode_evidence(d_bits, None, codes)
            if not ev.get("ok"):
                continue
            improvement = base_stage.get("distance_per_bit", 1.0) - ev["distance_per_bit"]
            cand = {"geometry": g["label"] + ("" if off == 0 else f" @ offset {off}"),
                    "geometry_label": g["label"],
                    "kind": g["kind"], "params": dict(g["params"]), "align_offset_bits": int(off),
                    "code": ev["code"], "distance_per_bit": ev["distance_per_bit"],
                    "z_score": ev.get("z_score"), "improvement_per_bit": float(improvement),
                    "relative_improvement": float(improvement /
                                                  max(base_stage.get("distance_per_bit", 1.0), 1e-9)),
                    "n_bits_ranked": int(stage_use)}
            if best_for_geometry is None or cand["distance_per_bit"] < best_for_geometry["distance_per_bit"]:
                best_for_geometry = cand
        if best_for_geometry is not None:
            results.append(best_for_geometry)
    results.sort(key=lambda r: r["distance_per_bit"])
    # --- stage 2: re-measure the leading candidates on the full stream -----------------------
    promoted = []
    for cand in results[:max(0, int(final_geometries))]:
        if n_use <= stage_use:
            break
        g = geom_by_label.get(cand.get("geometry_label"))
        if g is None:
            continue
        off = int(cand.get("align_offset_bits") or 0)
        ev = _best_decode_evidence(_aligned(bits[:n_use], g, off), None, codes)
        if not ev.get("ok"):
            continue
        imp = base.get("distance_per_bit", 1.0) - ev["distance_per_bit"]
        cand.update({"distance_per_bit": ev["distance_per_bit"], "code": ev["code"],
                     "z_score": ev.get("z_score"), "improvement_per_bit": float(imp),
                     "relative_improvement": float(imp /
                                                   max(base.get("distance_per_bit", 1.0), 1e-9)),
                     "n_bits_ranked": int(n_use), "verified_on_full_stream": True})
        promoted.append(cand)
    results.sort(key=lambda r: r["distance_per_bit"])
    # --- control group -------------------------------------------------------------------
    # Choosing the best of N geometries on *any* stream produces a positive improvement by chance
    # (measured: an uncoded BPSK record "improved" by 1-3 % with a block interleaver).  The same
    # test is therefore run on random permutations of the same bits: a geometry only counts as
    # evidence if it beats what a random reordering achieves.
    control = []
    if control_permutations > 0:
        rng = np.random.default_rng(seed)
        for _ in range(int(control_permutations)):
            if time_budget_s and (time.time() - t_start) > float(time_budget_s) + 10.0:
                truncated = True
                break
            perm = rng.permutation(n_use)
            ev = _best_decode_evidence(bits[:n_use][perm], None, codes)
            if not ev.get("ok"):
                continue
            imp = float(base.get("distance_per_bit", 1.0) - ev["distance_per_bit"])
            control.append({"geometry": "random control permutation", "kind": "control",
                            "params": {"seed": int(seed)},
                            "code": ev["code"], "distance_per_bit": ev["distance_per_bit"],
                            "z_score": ev.get("z_score"), "improvement_per_bit": imp,
                            "relative_improvement": float(imp / max(base.get("distance_per_bit", 1.0),
                                                                    1e-9))})
    ctrl_imp = np.asarray([c["relative_improvement"] for c in control], dtype=np.float64)
    ctrl = {"n": int(ctrl_imp.size),
            "mean_relative_improvement": float(ctrl_imp.mean()) if ctrl_imp.size else None,
            "std_relative_improvement": float(ctrl_imp.std(ddof=1)) if ctrl_imp.size > 1 else None,
            "max_relative_improvement": float(ctrl_imp.max()) if ctrl_imp.size else None}
    best = results[0] if results else None
    if best is not None and ctrl_imp.size:
        sd = float(ctrl["std_relative_improvement"] or 0.0)
        z = ((best["relative_improvement"] - float(ctrl["mean_relative_improvement"])) /
             max(sd, 1e-4))
        best = dict(best)
        best["control"] = ctrl
        best["control_z"] = float(z)
        best["beats_control_max"] = bool(best["relative_improvement"] > ctrl_imp.max())
    return {"ok": True, "baseline": base, "baseline_short": base_stage, "candidates": results,
            "best": best, "promoted_candidates": len(promoted),
            "control": ctrl, "control_cases": control, "n_bits_tested": n_use,
            "stage_bits": int(stage_use), "truncated": bool(truncated),
            "geometries_tested": int(tested), "geometries_total": int(len(geoms)),
            "time_budget_s": time_budget_s, "duration_s": round(time.time() - t_start, 2),
            "method": "apply each candidate de-interleaver, then Viterbi-decode with the standard "
                      "convolutional codes and compare the distance to the nearest codeword; the "
                      "same test is run on random permutations of the same bits as a control group"}


def _perm_for(n: int, geom: dict) -> np.ndarray:
    from .synth import interleave_deindices
    return interleave_deindices(n, geom["kind"], **geom["params"])


def _best_decode_evidence(bits: np.ndarray, soft: np.ndarray | None, codes: list) -> dict:
    """Hard-decision Viterbi distance per bit for each candidate code.

    Hard decisions are used deliberately: the metric is then a per-bit error fraction in [0, 0.5]
    with 0.5 as the exact value for a random stream, which makes both the baseline and the random
    permutation control directly interpretable.  Soft metrics are *not* on the same scale (they are
    negative log-likelihood sums) and comparing them across permutations is meaningless.
    """
    best = {"ok": False, "distance_per_bit": 1.0}
    for code in codes:
        cd = {"polys": list(code.polys), "K": code.K, "rate": code.n,
              "name": f"conv_K{code.K}_r1_{code.n}"}
        d = fec_mod.viterbi_decode(bits, cd, soft=None, terminate=False)
        if not d.get("ok"):
            continue
        if d["metric_per_bit"] < best["distance_per_bit"]:
            best = {"ok": True, "distance_per_bit": float(d["metric_per_bit"]), "code": cd["name"]}
    return best


def apply_deinterleave(bits: np.ndarray, soft: np.ndarray | None, kind: str, **kw) -> dict:
    """Apply one de-interleaver and report the before/after reliability structure."""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    out = {"bits": deinterleave(bits, kind, **kw), "kind": kind, "params": kw}
    if soft is not None:
        s = np.asarray(soft, dtype=np.float64).ravel()
        before = clustering_test(s)
        after = clustering_test(s[_perm_for(s.size, {"kind": kind, "params": kw})])
        out["soft"] = s[_perm_for(s.size, {"kind": kind, "params": kw})]
        out["clustering_before"] = before
        out["clustering_after"] = after
        out["dispersion_change"] = (float(before.get("dispersion_index", 1.0)) -
                                    float(after.get("dispersion_index", 1.0)))
    return out


def analyse_interleaving(soft: np.ndarray, bits: np.ndarray | None = None,
                         fec_result: dict | None = None, max_bits: int = 8000,
                         run_decode_test: bool = True, cancelled=None,
                         time_budget_s: float | None = 25.0) -> dict:
    """Full interleaving analysis: clustering test plus the de-interleaving decode test."""
    soft = None if soft is None else np.asarray(soft, dtype=np.float64).ravel()
    if bits is None and soft is not None:
        bits = (soft < 0).astype(np.int8)
    if bits is None:
        return {"ok": False, "hypotheses": [], "message": "no demodulated bits available"}
    bits = np.asarray(bits, dtype=np.int8).ravel()
    notes, hypotheses = [], []
    if bits.size < 128:
        return {"ok": False, "hypotheses": [],
                "message": f"only {bits.size} bits available; interleaver detection needs more",
                "notes": ["an interleaver permutation cannot be inferred from a handful of bits"]}

    cluster = clustering_test(soft) if soft is not None and soft.size >= 128 else {
        "ok": False, "message": "no soft decisions available (hard decisions only)"}
    if cluster.get("ok"):
        z = cluster["z_score"]
        ev = [f"unreliable-symbol dispersion index = {cluster['dispersion_index']:.2f} "
              f"(1.00 = memoryless) over {cluster['n_windows']} windows -> z = {z:+.1f}",
              f"mean burst length {cluster['mean_run']:.2f} symbols, longest {cluster['max_run']}, "
              f"{cluster['n_runs']} bursts",
              f"{cluster['interpretation']}"]
        hypotheses.append({"family": "interleaving", "kind": None,
                           "hypothesis": ("indication: unreliable symbols look dispersed "
                                          "(consistent with an interleaver)" if z < 2 else
                                          "indication: unreliable symbols look clustered "
                                          "(consistent with an un-interleaved channel)"),
                           # This test measures channel memory, not a permutation: it cannot name a
                           # geometry and on a memoryless channel it has no power.  It is therefore
                           # reported as an indication (capped, and excluded from the ranking) rather
                           # than as the platform's answer.
                           "confidence": float(min(0.25, clamp01(abs(z) / 6.0) * 0.5)),
                           "rank_excluded": True,
                           "status": "low_confidence",
                           "evidence": ev,
                           "limitations": ["this test responds to channel memory, not to a specific "
                                           "permutation: it cannot name the interleaver geometry",
                                           "on a memoryless AWGN channel this test has no power"]})
    else:
        notes.append("hard decisions only: the error-clustering test needs soft decisions")

    decode_test = None
    conv_hyp = None
    if fec_result and fec_result.get("hypotheses"):
        conv_hyp = next((h for h in fec_result["hypotheses"]
                         if h["family"] == "convolutional" and h["code"]), None)
    if run_decode_test and bits.size >= 512 and not (cancelled and cancelled()):
        codes = None
        if conv_hyp:
            name = conv_hyp["code"]
            codes = [c for c in STANDARD_CONV_CODES if f"conv_K{c.K}_r1_{c.n}" == name] or None
        decode_test = test_deinterleaving(soft if soft is not None else bits.astype(float),
                                         bits=bits, codes=codes, max_bits=max_bits,
                                         time_budget_s=time_budget_s)
        if decode_test.get("ok"):
            base = decode_test["baseline"]
            best = decode_test["best"]
            notes.append(f"de-interleaving decode test: {len(decode_test['candidates'])} of "
                         f"{decode_test.get('geometries_total', len(decode_test['candidates']))} "
                         f"geometries evaluated in {decode_test.get('duration_s')} s, baseline "
                         f"distance {base.get('distance_per_bit', float('nan')):.4f}"
                         f" per bit" + (f", best {best['geometry']} -> "
                                        f"{best['distance_per_bit']:.4f}" if best else ""))
            if best and best["relative_improvement"] > 0.10 and best.get("beats_control_max"):
                ctrl = best.get("control") or {}
                z = float(best.get("control_z") or 0.0)
                conf = clamp01((z - 2.0) / 4.0) * clamp01(best["relative_improvement"] / 0.15)
                # Two evidence levels, as everywhere else in the platform: above 0.20 the geometry is
                # a *claim* (it is also what ``best`` returns); between 0.10 and 0.20 it is recorded as
                # an *indication* and excluded from the ranking, because a 10-15 % decode improvement
                # was measured to be reachable by a wrong geometry on a diagonal-interleaved record.
                reporting = float(best["relative_improvement"]) >= 0.20
                hypotheses.append({"family": "interleaving", "kind": best["kind"],
                                   "hypothesis": f"{best['geometry']} interleaver "
                                                 f"(de-interleaving improves the "
                                                 f"{best['code']} decode)",
                                   "confidence": float(conf),
                                   "relative_improvement": float(best["relative_improvement"]),
                                   "improvement_per_bit": float(best["improvement_per_bit"]),
                                   "align_offset_bits": int(best.get("align_offset_bits") or 0),
                                   "beats_control_max": bool(best.get("beats_control_max")),
                                   "control": best.get("control"),
                                   "control_z": best.get("control_z"),
                                   "status": ("ok" if (reporting and conf >= 0.4) else "low_confidence"),
                                   "rank_excluded": (not reporting),
                                   "evidence_level": ("claim" if reporting else "indication"),
                                   "evidence": [
                                       f"distance to the nearest {best['code']} codeword drops from "
                                       f"{base['distance_per_bit']:.4f} to "
                                       f"{best['distance_per_bit']:.4f} per bit "
                                       f"({best['relative_improvement'] * 100:.1f} % improvement) "
                                       f"after de-interleaving",
                                       f"{decode_test['n_bits_tested']:,} bits, the same decoder and "
                                       f"the same code used before and after",
                                       "the geometry was selected from the candidate catalogue "
                                       "purely by the measured decode improvement",
                                       f"control group: {ctrl.get('n')} random permutations of the "
                                       f"same bits improved the decode by "
                                       f"{(ctrl.get('mean_relative_improvement') or 0) * 100:.2f} % on "
                                       f"average (worst "
                                       f"{(ctrl.get('max_relative_improvement') or 0) * 100:.2f} %); "
                                       f"this geometry scores "
                                       f"{best['relative_improvement'] * 100:.2f} %, z={z:.1f}"],
                                   "limitations": ([
                                       "the geometry is the best fit among the tested catalogue; a "
                                       "different interleaver with a similar dispersion could score "
                                       "similarly",
                                       "the decoded bits themselves are not verified"] +
                                       ([] if reporting else [
                                           "the measured improvement is above the 0.10 indication "
                                           "level but below the 0.20 reporting threshold, so this is "
                                           "an indication and not a claim"]))})
                if not reporting:
                    notes.append(f"the best geometry ({best['geometry']}) improves the decoder by "
                                 f"{best['relative_improvement'] * 100:.1f} %: above the 0.10 "
                                 f"indication level but below the 0.20 reporting threshold, so it is "
                                 f"listed as an indication and no interleaver is claimed")
            elif not hypotheses:
                if best and not best.get("beats_control_max"):
                    notes.append("the best geometry did not beat random permutations of the same "
                                 "bits, so it is chance rather than evidence: no interleaver is "
                                 "claimed")
                else:
                    notes.append("the decode test found no geometry that improves the decoder, so no "
                                 "interleaver is claimed: the stream may be un-interleaved, or the "
                                 "channel/decoder never had enough evidence for the test to work")
    elif not run_decode_test:
        notes.append("the de-interleaving decode test was disabled for this run")

    hypotheses.sort(key=lambda h: -h["confidence"])
    ranked = [h for h in hypotheses if not h.get("rank_excluded")]
    ranked.sort(key=lambda h: -(h.get("confidence") or 0.0))
    best = ranked[0] if ranked else None
    if best is not None and best.get("confidence", 0.0) < 0.2:
        # same evidence rule as the FEC analyser: below the reporting threshold nothing is claimed,
        # although the full candidate list is still returned for inspection
        notes.append(f"the strongest interleaver hypothesis ({best.get('kind')}, confidence "
                     f"{best['confidence']:.2f}) is below the 0.20 reporting threshold and is not "
                     "claimed")
        best = None
    if best is not None and len(ranked) > 1:
        r2 = ranked[1]
        gap = float(best.get("relative_improvement") or 0.0) - float(r2.get("relative_improvement") or 0.0)
        if best.get("kind") != r2.get("kind") and gap < 0.05:
            notes.append(f"the leading geometries are not separated by the measurement "
                         f"({best.get('kind')} {float(best.get('relative_improvement') or 0) * 100:.1f} % "
                         f"versus {r2.get('kind')} "
                         f"{float(r2.get('relative_improvement') or 0) * 100:.1f} %): the reported "
                         f"geometry is the best fit among the tested catalogue, not a certainty")
    if best is None:
        notes.append("no reliable interleaving pattern was detected for this record")
    return {"ok": True, "hypotheses": hypotheses, "best": best,
            "clustering": cluster if cluster.get("ok") else None,
            "decode_test": decode_test, "notes": notes,
            "method": "unreliable-symbol dispersion test on the soft decisions plus a "
                      "de-interleave-and-decode improvement test over the interleaver catalogue",
            "n_bits": int(bits.size)}


def interleaving_param(result: dict) -> dict:
    best = (result or {}).get("best")
    if not best or best.get("confidence", 0.0) < 0.2:
        return unavailable("Interleaving", "no interleaver could be evidenced for this record",
                           method="dispersion test + de-interleave/decode improvement test")
    return param("Interleaving", best["hypothesis"], None, best["confidence"],
                 "best-scoring interleaver hypothesis", status=best.get("status", "ok"),
                 evidence=best.get("evidence", [])[:3], limitations=best.get("limitations", [])[:2])
