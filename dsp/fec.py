"""Automatic FEC detection and decoding - hypothesis testing with measurable evidence.

The platform never claims "FEC: K=7 rate 1/2" from a hunch.  Each candidate code is *tested* against
the demodulated bits and against a null hypothesis:

* **Convolutional codes** - a hard-decision Viterbi decoder runs for every standard code in
  :data:`dsp.synth.STANDARD_CONV_CODES`.  Its path metric, expressed per received bit, is the
  Hamming distance from the observed bits to the code's nearest codeword.  The same statistic is
  measured on random data (the null distribution), so the evidence is a z-score.  Random uncoded
  data sits at the null mean; a genuinely coded stream sits far below it.
* **Reed-Solomon** - GF(256) syndromes are evaluated for every plausible codeword alignment; a
  Berlekamp-Massey error locator is searched, the error positions are found by a Chien search and
  the magnitudes are solved from the syndromes.  A candidate whose syndromes all vanish (or whose
  correction zeroes them) is a *proof* of that code for that block; random bytes essentially never
  satisfy it.
* **LDPC** - blind identification from a bitstream alone is not possible (the parity-check matrix
  cannot be recovered), so the platform says so and reports only a weak linear-structure statistic.
  No LDPC claim is ever made without a supplied matrix.
* **Concatenated** - reported only when two independent families each pass their own evidence test.

Nothing here uses ground truth: only the demodulated bits (and optionally their soft values).
"""

from __future__ import annotations

import numpy as np

from .synth import STANDARD_CONV_CODES, ConvCode, conv_encode
from .utils import clamp01, param, unavailable

__all__ = ["viterbi_decode", "convolutional_hypotheses", "conv_hypothesis_record",
           "apply_conv_decode", "rs_encode", "rs_syndrome", "rs_decode_block",
           "reed_solomon_hypotheses", "rs_decode_bytes", "ldpc_hypotheses", "analyse_fec"]


# =======================================================================================
# GF(256) arithmetic
# =======================================================================================
def _build_gf(prim: int = 0x11D):
    exp = np.zeros(512, dtype=np.int32)
    log = np.zeros(256, dtype=np.int32)
    x = 1
    for i in range(255):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & 0x100:
            x ^= prim
    for i in range(255, 512):
        exp[i] = exp[i - 255]
    mul = np.zeros((256, 256), dtype=np.uint8)
    for a in range(256):
        if a == 0:
            continue
        la = log[a]
        for b in range(256):
            if b == 0:
                continue
            mul[a, b] = exp[(la + log[b]) % 255]
    mul_inv = np.zeros(256, dtype=np.uint8)
    for a in range(1, 256):
        mul_inv[a] = exp[(255 - log[a]) % 255]
    mul_inv[0] = 0
    return exp, log, mul, mul_inv


GF_EXP, GF_LOG, GF_MUL, GF_INV = _build_gf()


def _gf_poly_eval_asc(poly: np.ndarray, x: int) -> int:
    """Evaluate a polynomial stored with index 0 = the constant term (BM locator order)."""
    acc = 0
    for c in np.asarray(poly)[::-1]:
        acc = _gmul(acc, x) ^ int(c)
    return acc


def _gmul(a, b):
    return int(GF_MUL[int(a) & 0xFF, int(b) & 0xFF])


def _gf_poly_eval(poly: np.ndarray, x: int) -> int:
    """Evaluate a polynomial whose index 0 is the *highest* power (transmission order)."""
    acc = 0
    for c in np.asarray(poly):
        acc = _gmul(acc, x) ^ int(c)
    return acc


def _gf_poly_div(dividend: np.ndarray, divisor: np.ndarray) -> np.ndarray:
    """Polynomial division over GF(256); returns the remainder."""
    d = np.asarray(dividend, dtype=np.int32).copy()
    q = np.asarray(divisor, dtype=np.int32)
    if q.size == 0 or q[0] == 0:
        raise ValueError("divisor must have a non-zero leading coefficient")
    for i in range(d.size - q.size + 1):
        if d[i] == 0:
            continue
        coef = _gmul(int(d[i]), int(GF_INV[int(q[0])]))
        for j in range(q.size):
            d[i + j] ^= _gmul(coef, int(q[j]))
    return d[-(q.size - 1):] if q.size > 1 else np.zeros(0, dtype=np.int32)


def rs_generator(nsym: int, fcr: int = 1) -> np.ndarray:
    """Generator polynomial g(x) = prod (x - a^(fcr+i))."""
    g = np.array([1], dtype=np.int32)
    for i in range(nsym):
        root = int(GF_EXP[(fcr + i) % 255])
        new = np.zeros(g.size + 1, dtype=np.int32)
        new[:-1] ^= g
        new[1:] ^= np.array([_gmul(int(c), root) for c in g], dtype=np.int32)
        g = new
    return g


def rs_encode(msg: np.ndarray, nsym: int, fcr: int = 1) -> np.ndarray:
    """Systematic RS encode: ``msg`` bytes followed by ``nsym`` parity bytes."""
    msg = np.asarray(msg, dtype=np.uint8).ravel()
    g = rs_generator(nsym, fcr)
    padded = np.concatenate([msg.astype(np.int32), np.zeros(nsym, dtype=np.int32)])
    par = _gf_poly_div(padded, g)
    return np.concatenate([msg, par.astype(np.uint8)])


def rs_syndrome(block: np.ndarray, nsym: int, fcr: int = 1) -> np.ndarray:
    """Syndromes S_j = C(a^(fcr+j)) of a received word (first byte = highest power)."""
    block = np.asarray(block, dtype=np.int32)
    return np.array([_gf_poly_eval(block, int(GF_EXP[(fcr + j) % 255])) for j in range(nsym)],
                    dtype=np.int32)


def _bm_locator(S: np.ndarray) -> np.ndarray | None:
    """Berlekamp-Massey: error-locator polynomial from the syndromes (None if not decodable)."""
    nsym = S.size
    C = np.zeros(nsym + 1, dtype=np.int32)
    B = np.zeros(nsym + 1, dtype=np.int32)
    C[0] = 1
    B[0] = 1
    L, m, b = 0, 1, 1
    for n in range(nsym):
        d = int(S[n])
        for i in range(1, L + 1):
            d ^= _gmul(int(C[i]), int(S[n - i]))
        if d == 0:
            m += 1
            continue
        T = C.copy()
        coef = _gmul(d, int(GF_INV[b]))
        for i in range(nsym + 1 - m):
            if B[i]:
                C[i + m] ^= _gmul(coef, int(B[i]))
        if 2 * L <= n:
            L = n + 1 - L
            B = T
            b = d
            m = 1
        else:
            m += 1
    if L == 0 or L > nsym // 2:
        return None
    return C[: L + 1]


def _chien_roots(locator: np.ndarray, n: int) -> list[int]:
    """Chien search: error positions p such that the locator vanishes at a^(p-(n-1))."""
    roots = []
    for i in range(255):
        if _gf_poly_eval_asc(locator, int(GF_EXP[i])) == 0:
            p = (i + n - 1) % 255
            if p < n:
                roots.append(int(p))
    return roots


def _solve_magnitudes(positions: list[int], S: np.ndarray, n: int, fcr: int = 1) -> list[int]:
    """Solve the Vandermonde system S_j = sum e_p X_p^(fcr+j) for the error magnitudes."""
    L = len(positions)
    if L == 0:
        return []
    X = np.array([int(GF_EXP[(n - 1 - p) % 255]) for p in positions], dtype=np.int32)
    A = np.zeros((L, L), dtype=np.int32)
    for j in range(L):
        for i in range(L):
            A[j, i] = int(GF_EXP[((fcr + j) * int(GF_LOG[X[i]])) % 255])
    bvec = np.array([int(S[j]) for j in range(L)], dtype=np.int32)
    # Gaussian elimination over GF(256)
    for c in range(L):
        piv = None
        for r in range(c, L):
            if A[r, c]:
                piv = r
                break
        if piv is None:
            return []
        if piv != c:
            A[[c, piv]] = A[[piv, c]]
            bvec[[c, piv]] = bvec[[piv, c]]
        inv = int(GF_INV[A[c, c]])
        A[c] = [_gmul(inv, int(v)) for v in A[c]]
        bvec[c] = _gmul(inv, int(bvec[c]))
        for r in range(L):
            if r != c and A[r, c]:
                f = int(A[r, c])
                A[r] = [int(A[r, k]) ^ _gmul(f, int(A[c, k])) for k in range(L)]
                bvec[r] = int(bvec[r]) ^ _gmul(f, int(bvec[c]))
    return [int(v) for v in bvec]


def rs_decode_block(block: np.ndarray, nsym: int = 32, fcr: int = 1) -> dict:
    """Decode one RS codeword; report whether the syndromes are zeroed by the correction."""
    block = np.asarray(block, dtype=np.uint8).astype(np.int32)
    n = block.size
    S = rs_syndrome(block, nsym, fcr=fcr)
    if not np.any(S):
        return {"ok": True, "errors_found": 0, "corrected": block.astype(np.uint8), "positions": [],
                "syndromes_zero": True, "decoded": True, "message": None}
    loc = _bm_locator(S)
    if loc is None:
        return {"ok": False, "errors_found": None, "corrected": None, "positions": [],
                "syndromes_zero": False, "decoded": False,
                "message": f"no error locator of degree <= {nsym // 2}: the block is not a "
                           f"correctable RS({n},{n - nsym}) word"}
    pos = _chien_roots(loc, n)
    if len(pos) != loc.size - 1:
        return {"ok": False, "errors_found": len(pos), "corrected": None, "positions": pos,
                "syndromes_zero": False, "decoded": False,
                "message": "the error-locator degree does not match the roots found"}
    mags = _solve_magnitudes(pos, S, n, fcr=fcr)
    if len(mags) != len(pos):
        return {"ok": False, "errors_found": len(pos), "corrected": None, "positions": pos,
                "syndromes_zero": False, "decoded": False,
                "message": "error magnitudes could not be solved (the locator is spurious)"}
    corr = block.copy()
    for p, e in zip(pos, mags):
        corr[p] ^= int(e)
    ok = not np.any(rs_syndrome(corr, nsym, fcr=fcr))
    return {"ok": bool(ok), "errors_found": int(len(pos)), "corrected": corr.astype(np.uint8),
            "positions": pos, "syndromes_zero": bool(ok), "decoded": bool(ok),
            "message": None if ok else "correction did not zero the syndromes"}


# =======================================================================================
# Convolutional codes
# =======================================================================================
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _tables(code: dict):
    """(output bits per (state,input), predecessor states) for a rate-1/n code."""
    g = [int(p) for p in code["polys"]]
    K = int(code.get("K") or code.get("constraint"))
    rate = len(g)
    n_states = 1 << (K - 1)
    mask = n_states - 1
    out_bits = np.zeros((n_states, 2), dtype=np.int32)
    for st in range(n_states):
        for b in (0, 1):
            reg = ((st << 1) | b) & ((1 << K) - 1)
            bits = 0
            for j, p in enumerate(g):
                bits |= ((bin(reg & p).count("1") & 1) << j)
            out_bits[st, b] = bits
    prev0 = (np.arange(n_states) >> 1).astype(np.int32)                    # dropped bit = 0
    prev1 = (prev0 | (1 << (K - 2))).astype(np.int32)                      # dropped bit = 1
    return out_bits, prev0, prev1, rate, n_states, mask


def viterbi_decode(bits: np.ndarray, code: dict, soft: np.ndarray | None = None,
                   terminate: bool = True) -> dict:
    """Vectorised Viterbi decoder for a rate-1/n convolutional code.

    Hard decisions use Hamming distance, soft decisions the LLR branch cost.  The state update is
    the standard shift-register trellis; the information bit of a step is the LSB of the state
    that step ends in, which is what the traceback reads out.
    """
    bits = np.asarray(bits, dtype=np.int8).ravel()
    out_bits, prev0, prev1, rate, n_states, mask = _tables(code)
    if bits.size < 4 * rate:
        return {"ok": False, "message": "not enough bits for a Viterbi decode"}
    n_steps = bits.size // rate
    obs = bits[: n_steps * rate].reshape(n_steps, rate).astype(np.int32)
    obs_int = np.zeros(n_steps, dtype=np.int32)
    for j in range(rate):
        obs_int |= (obs[:, j] & 1) << j
    llr = None
    if soft is not None and np.asarray(soft).size >= n_steps * rate:
        llr = np.asarray(soft, dtype=np.float64).ravel()[: n_steps * rate].reshape(n_steps, rate)
    ns = np.arange(n_states, dtype=np.int32)
    b_in = ns & 1
    BIG = 1e18
    metrics = np.full(n_states, BIG)
    metrics[0] = 0.0
    trace = np.zeros((n_steps, n_states), dtype=np.uint8)
    for t in range(n_steps):
        if llr is None:
            xor = (out_bits ^ obs_int[t]) & 0xFF
            bm = _POPCOUNT[xor].astype(np.float64)                  # (n_states, 2)
        else:
            row = llr[t]
            exp0 = (out_bits >> 0) & 1
            bm = np.zeros((n_states, 2))
            for j in range(rate):
                e = (out_bits >> j) & 1
                bm += np.where(e == 1, -row[j], row[j])
        m0 = metrics[prev0] + bm[prev0, b_in]
        m1 = metrics[prev1] + bm[prev1, b_in]
        use0 = m0 <= m1
        trace[t] = (~use0).astype(np.uint8)
        metrics = np.where(use0, m0, m1)
    st = 0 if terminate else int(np.argmin(metrics))
    final_metric = float(metrics[st])
    decoded = np.zeros(n_steps, dtype=np.int8)
    for t in range(n_steps - 1, -1, -1):
        decoded[t] = st & 1
        st = int(prev0[st]) if trace[t, st] == 0 else int(prev1[st])
    return {"ok": True, "information_bits": decoded, "metric": final_metric,
            "metric_per_bit": float(final_metric / max(n_steps * rate, 1)),
            "n_steps": int(n_steps), "code": code.get("name", "unknown"), "rate": rate,
            "soft": llr is not None, "n_states": n_states}


_NULL_CACHE: dict = {}


def _null_distance_fraction(code: dict, n_bits: int, trials: int = 6, seed: int = 17) -> dict:
    """Null distribution of the Viterbi distance for random (uncoded) bits of the same length.

    The result depends only on the code and the bit count, so it is cached: the same null can
    legitimately be reused across analyses - it is a property of the code, not of the data.
    """
    key = (code.get("name"), int(n_bits), int(trials))
    if key in _NULL_CACHE:
        return _NULL_CACHE[key]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(trials):
        r = rng.integers(0, 2, size=n_bits).astype(np.int8)
        d = viterbi_decode(r, code, terminate=False)
        if d.get("ok"):
            vals.append(d["metric_per_bit"])
    if not vals:
        return {"ok": False}
    out = {"ok": True, "mean": float(np.mean(vals)), "std": max(float(np.std(vals)), 1e-3),
           "n_trials": len(vals), "values": [round(float(v), 4) for v in vals],
           "cached": False}
    _NULL_CACHE[key] = out
    return out


def convolutional_hypotheses(bits: np.ndarray, codes: list | None = None, n_bits: int = 6000,
                             null_trials: int = 6, max_codes: int = 8) -> list[dict]:
    """Test every standard convolutional code (from :data:`dsp.synth.STANDARD_CONV_CODES`)."""
    codes = codes if codes is not None else list(STANDARD_CONV_CODES)
    bits = np.asarray(bits, dtype=np.int8).ravel()
    out = []
    for i, code in enumerate(codes):
        if i >= max_codes:
            break
        name = f"conv_K{code.K}_r1_{code.n}"
        cd = {"polys": list(code.polys), "K": code.K, "rate": code.n, "name": name}
        n_use = int(min(bits.size, n_bits))
        n_use -= n_use % cd["rate"]
        if n_use < 64:
            continue
        seg = bits[:n_use]
        d = viterbi_decode(seg, cd, terminate=False)
        if not d.get("ok"):
            continue
        null = _null_distance_fraction(cd, n_use, trials=null_trials, seed=1000 + i)
        if not null.get("ok"):
            continue
        z = (null["mean"] - d["metric_per_bit"]) / null["std"]
        reenc = conv_encode(d["information_bits"], code)
        m = min(reenc.size, seg.size)
        mismatch = float(np.mean(reenc[:m] != seg[:m])) if m else 1.0
        out.append({"family": "convolutional", "name": name, "rate": cd["rate"],
                    "constraint": cd["K"], "polys": cd["polys"],
                    "distance_per_bit": d["metric_per_bit"], "null_mean": null["mean"],
                    "null_std": null["std"], "z_score": float(z),
                    "reencode_mismatch": mismatch, "n_bits_tested": int(n_use),
                    "null_values": null["values"],
                    "hypothesis": f"{name} (rate 1/{cd['rate']}, K={cd['K']})"})
    out.sort(key=lambda r: -r["z_score"])
    return out


def conv_hypothesis_record(hyp: dict, n_bits_total: int) -> dict:
    z = hyp["z_score"]
    conf = clamp01(min(1.0, max(0.0, (z - 3.0) / 12.0)) *
                   (1.0 if hyp["reencode_mismatch"] < 0.12 else 0.4))
    status = "ok" if conf >= 0.25 else ("low_confidence" if conf > 0.05 else "unable")
    ev = [f"Viterbi distance to the nearest codeword: {hyp['distance_per_bit']:.4f} per received "
          f"bit ({hyp['n_bits_tested']:,} bits tested)",
          f"null hypothesis, random uncoded bits through the same decoder: "
          f"{hyp['null_mean']:.4f} +/- {hyp['null_std']:.4f} -> z = {z:+.1f}",
          f"re-encoding the decoded information bits reproduces "
          f"{(1 - hyp['reencode_mismatch']) * 100:.2f} % of the received bits"]
    lim = []
    if conf < 0.25:
        lim.append("the distance is not significantly below the random-data null: this code cannot "
                   "be claimed")
    if n_bits_total < 512:
        lim.append("fewer than 512 bits were available, so the test has little statistical power")
    if hyp["rate"] > 1:
        lim.append(f"the rate-1/{hyp['rate']} hypothesis was also tested against rate-1/2 codes; the "
                   "reported one is the best fit of the tested family")
    return {"family": "convolutional", "hypothesis": hyp["hypothesis"], "code": hyp["name"],
            "rate": hyp["rate"], "constraint": hyp.get("constraint"),
            "z_score": float(z), "confidence": float(conf), "status": status,
            "evidence": ev, "limitations": lim, "metrics": hyp}


def _find_code(code_name: str) -> ConvCode | None:
    for c in STANDARD_CONV_CODES:
        if f"conv_K{c.K}_r1_{c.n}" == code_name:
            return c
    return None


def apply_conv_decode(bits: np.ndarray, soft: np.ndarray | None, code_name: str,
                      max_bits: int = 200000) -> dict:
    code = _find_code(code_name)
    if code is None:
        return {"ok": False, "message": f"unknown convolutional code '{code_name}'"}
    cd = {"polys": list(code.polys), "K": code.K, "rate": code.n, "name": code_name}
    bits = np.asarray(bits, dtype=np.int8).ravel()[:max_bits]
    soft_use = None if soft is None else np.asarray(soft, dtype=np.float64).ravel()[:max_bits]
    d = viterbi_decode(bits, cd, soft=soft_use, terminate=False)
    if not d.get("ok"):
        return d
    info = d["information_bits"]
    reenc = conv_encode(info, code)
    m = min(reenc.size, bits.size)
    residual = float(np.mean(reenc[:m] != bits[:m])) if m else None
    return {"ok": True, "code": code_name, "rate": code.n, "constraint": code.K,
            "information_bits": info, "n_information_bits": int(info.size),
            "n_code_bits": int(bits.size), "soft_decisions": bool(soft is not None),
            "distance_per_bit_before": float(d["metric_per_bit"]),
            "residual_mismatch_after": residual,
            "note": "the residual figure is the mismatch between the re-encoded decode and the "
                    "received bits; it measures what the decoder attributes to the channel and is "
                    "not a verified error count"}


# =======================================================================================
# Reed-Solomon hypothesis testing
# =======================================================================================
def _rs_syndrome_matrix(blocks: np.ndarray, nsym: int, fcr: int = 1) -> np.ndarray:
    """Vectorised syndromes for a stack of blocks: S[j, m] = C_m(a^(fcr+j))."""
    B = np.asarray(blocks, dtype=np.int32)
    m, n = B.shape
    j = np.arange(nsym)[:, None]
    i = np.arange(n)[None, :]
    X = GF_EXP[(((fcr + j) % 255) * ((n - 1 - i) % 255)) % 255]        # (nsym, n)
    prod = GF_MUL[B[None, :, :], X[:, None, :]]                        # (nsym, m, n)
    return np.bitwise_xor.reduce(prod.astype(np.int32), axis=2)        # (nsym, m)


def _rs_null_valid_fraction(n: int, nsym: int, n_blocks: int, trials: int = 3,
                            seed: int = 11) -> dict:
    """Measure the null hypothesis: how often random bytes pass the same RS test."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(trials):
        data = rng.integers(0, 256, size=n * n_blocks, dtype=np.uint8)
        blocks = data.reshape(n_blocks, n)
        S = _rs_syndrome_matrix(blocks, nsym)
        clean = int(np.count_nonzero(~np.any(S, axis=0)))
        corrected = 0
        for b in range(n_blocks):
            if np.any(S[:, b]):
                d = rs_decode_block(blocks[b], nsym)
                if d.get("ok"):
                    corrected += 1
        vals.append((clean + corrected) / n_blocks)
    return {"ok": True, "mean": float(np.mean(vals)), "std": max(float(np.std(vals)), 0.02),
            "n_trials": len(vals), "values": [round(float(v), 3) for v in vals],
            "n_blocks_per_trial": int(n_blocks)}


def reed_solomon_hypotheses(data: np.ndarray, candidates: list[tuple[int, int]] | None = None,
                            max_blocks: int = 8, max_phase: int = 48,
                            null_trials: int = 3) -> list[dict]:
    """Test RS(n,k) hypotheses over plausible codeword alignments, with a measured null.

    The codeword phase is found by counting blocks whose GF(256) syndromes vanish at each
    alignment (a cheap vectorised test); the Berlekamp-Massey/Chien decode then runs only for the
    best alignments.  Every candidate is compared against the same statistic measured on random
    bytes of the same size, so weak configurations with little parity cannot score.
    """
    data = np.asarray(data, dtype=np.uint8).ravel()
    candidates = candidates or [(255, 32), (204, 16), (255, 16), (255, 8), (204, 8), (255, 4)]
    out = []
    for (n, nsym) in candidates:
        if data.size < n * 2:
            continue
        best = None
        for phase in range(0, min(max_phase, n)):
            starts = np.arange(phase, phase + max_blocks * n, n)
            starts = starts[starts + n <= data.size]
            if starts.size < 2:
                break
            blocks = np.stack([data[i:i + n] for i in starts])
            S = _rs_syndrome_matrix(blocks, nsym)
            clean = int(np.count_nonzero(~np.any(S, axis=0)))
            if best is None or clean > best["clean"]:
                best = {"clean": clean, "phase": int(phase), "blocks": blocks,
                        "nonzero_idx": [int(i) for i in np.flatnonzero(np.any(S, axis=0))]}
        if best is None:
            continue
        n_corr, n_uncorr, n_small = 0, 0, 0
        for idx in best["nonzero_idx"]:
            d = rs_decode_block(best["blocks"][idx], nsym)
            if d.get("ok"):
                n_corr += 1
                if (d.get("errors_found") or 0) <= max(1, nsym // 4):
                    n_small += 1
            else:
                n_uncorr += 1
        n_blocks = int(best["blocks"].shape[0])
        valid = (best["clean"] + n_corr) / max(n_blocks, 1)
        null = _rs_null_valid_fraction(n, nsym, n_blocks, trials=null_trials, seed=7 + nsym)
        z = (valid - null["mean"]) / max(null["std"], 0.02)
        out.append({"family": "reed-solomon", "n": int(n), "nsym": int(nsym), "k": int(n - nsym),
                    "t": int(nsym // 2), "phase": int(best["phase"]),
                    "blocks_tested": n_blocks, "blocks_clean": int(best["clean"]),
                    "blocks_corrected": int(n_corr), "blocks_uncorrectable": int(n_uncorr),
                    "clean_corrections": int(n_small), "valid_fraction": float(valid),
                    "null_valid_fraction": float(null["mean"]), "null_std": float(null["std"]),
                    "z_score": float(z), "null_values": null["values"],
                    "random_valid_fraction": float(255.0 ** (-max(1, nsym // 2))),
                    "hypothesis": f"Reed-Solomon({n},{n - nsym}) over GF(256), t={nsym // 2}"})
    out.sort(key=lambda r: (-r["z_score"], -r["nsym"]))
    return out


def rs_decode_bytes(data: np.ndarray, n: int, nsym: int, phase: int = 0,
                    max_blocks: int = 4096) -> dict:
    data = np.asarray(data, dtype=np.uint8).ravel()
    out, stats = [], {"blocks": 0, "clean": 0, "corrected": 0, "uncorrectable": 0,
                      "symbols_corrected": 0}
    i = int(phase)
    while i + n <= data.size and stats["blocks"] < max_blocks:
        blk = data[i:i + n]
        d = rs_decode_block(blk, nsym)
        stats["blocks"] += 1
        if d.get("syndromes_zero"):
            stats["clean"] += 1
            out.append(blk[: n - nsym])
        elif d.get("ok"):
            stats["corrected"] += 1
            stats["symbols_corrected"] += int(d.get("errors_found") or 0)
            out.append(d["corrected"][: n - nsym])
        else:
            stats["uncorrectable"] += 1
            out.append(blk[: n - nsym])
        i += n
    payload = np.concatenate(out) if out else np.zeros(0, dtype=np.uint8)
    return {"ok": bool(stats["blocks"]), "payload": payload, "stats": stats,
            "payload_bytes": int(payload.size),
            "note": "parity symbols are removed; uncorrectable blocks are passed through unchanged "
                    "and counted"}


# =======================================================================================
# LDPC (honest limits) and the top-level entry point
# =======================================================================================
def _linear_structure_statistic(bits: np.ndarray, n_bits: int = 20000, block: int = 64) -> dict:
    bits = np.asarray(bits, dtype=np.uint8).ravel()[:n_bits]
    nb = (bits.size // block) * block
    if nb < block * 8:
        return {"ok": False, "message": "not enough bits for a linear-structure test"}
    M = bits[:nb].reshape(-1, block)
    rng = np.random.default_rng(3)
    idx = rng.choice(M.shape[0], size=min(M.shape[0], block + 8), replace=False)
    A = M[idx].astype(np.uint8).copy()
    rows, cols, rank = A.shape[0], A.shape[1], 0
    for c in range(cols):
        piv = None
        for r in range(rank, rows):
            if A[r, c]:
                piv = r
                break
        if piv is None:
            continue
        A[[rank, piv]] = A[[piv, rank]]
        for r in range(rows):
            if r != rank and A[r, c]:
                A[r] ^= A[rank]
        rank += 1
    return {"ok": True, "rank": int(rank), "block": int(block), "rows": int(rows),
            "rank_deficiency": int(cols - rank),
            "note": "random unstructured bits give a full rank; a deficiency indicates short linear "
                    "relations between blocks"}


def ldpc_hypotheses(bits: np.ndarray, max_bits: int = 20000) -> list[dict]:
    stat = _linear_structure_statistic(bits, n_bits=max_bits)
    ev = []
    if stat.get("ok"):
        ev.append(f"GF(2) rank of {stat['rows']} stacked {stat['block']}-bit blocks: "
                  f"{stat['rank']}/{stat['block']} (deficiency {stat['rank_deficiency']})")
    return [{"family": "ldpc", "hypothesis": "LDPC / structured linear block code",
             "status": "unable", "confidence": 0.0,
             "evidence": ev + ["blind LDPC identification needs the parity-check matrix (or the "
                               "block length and degree distribution); it cannot be recovered from "
                               "a bitstream of this length and error rate"],
             "limitations": ["no LDPC claim is made without a supplied parity-check matrix",
                             "the rank statistic is weak evidence: it cannot separate a random "
                             "stream from most structured codes"],
             "metrics": stat}]


def analyse_fec(bits: np.ndarray, soft: np.ndarray | None = None, max_bits: int = 12000,
                null_trials: int = 6, try_rs: bool = True, cancelled=None) -> dict:
    """Test all supported FEC families against the demodulated bit stream."""
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if bits.size < 64:
        return {"ok": False, "hypotheses": [], "best": None,
                "message": f"only {bits.size} bits are available; FEC detection needs at least 64",
                "notes": ["FEC identification is impossible from a handful of bits"]}
    if cancelled and cancelled():
        return {"ok": False, "cancelled": True, "hypotheses": [], "best": None}
    # trailing flush bits and near-zero samples can bias the test; work on a power-of-two prefix
    notes, hypotheses = [], []
    conv = convolutional_hypotheses(bits, n_bits=max_bits, null_trials=null_trials)
    for h in conv[:4]:
        hypotheses.append(conv_hypothesis_record(h, bits.size))
    if cancelled and cancelled():
        return {"ok": False, "cancelled": True, "hypotheses": [], "best": None}
    if try_rs and bits.size >= 255 * 8 * 2:
        byte_stream = np.packbits(bits[: (bits.size // 8) * 8].astype(np.uint8)).astype(np.uint8)
        for h in reed_solomon_hypotheses(byte_stream)[:3]:
            z = h["z_score"]
            frac = h["valid_fraction"]
            # evidence = how far above the random-data null this alignment sits, how many blocks
            # actually validate, and how much parity the candidate carries
            n_valid = h["blocks_clean"] + h["blocks_corrected"]
            support = 0.15 if n_valid <= 1 else (0.6 if n_valid <= 3 else 1.0)
            conf = (clamp01((z - 1.5) / 6.0) * clamp01(0.25 + 0.75 * clamp01(h["nsym"] / 16.0))
                    * support)
            if h["blocks_clean"] == 0 and h["blocks_corrected"] < 2:
                conf = min(conf, 0.15)
            status = "ok" if conf >= 0.25 else ("low_confidence" if conf > 0.05 else "unable")
            ev = [f"{h['blocks_clean']}/{h['blocks_tested']} blocks have all-zero GF(256) syndromes "
                  f"and {h['blocks_corrected']} more are correctable, at codeword phase {h['phase']} "
                  f"(block length {h['n']}, {h['nsym']} parity symbols)",
                  f"measured null (random bytes of the same size through the same test): "
                  f"{h['null_valid_fraction']:.3f} +/- {h['null_std']:.3f} valid fraction "
                  f"-> z = {z:+.1f}",
                  f"binding lower bound from the parity count alone: {h['random_valid_fraction']:.2e}",
                  f"uncorrectable blocks: {h['blocks_uncorrectable']}"]
            lim = []
            if conf < 0.25:
                lim.append("the alignment is not significantly better than the random-data null, so "
                           "this RS configuration cannot be claimed")
            if h["blocks_tested"] < 4:
                lim.append("fewer than four blocks were available, so this is weak evidence")
            hypotheses.append({"family": "reed-solomon", "hypothesis": h["hypothesis"],
                               "code": f"RS({h['n']},{h['k']})", "nsym": h["nsym"], "t": h["t"],
                               "phase": h["phase"], "confidence": float(conf), "status": status,
                               "evidence": ev, "limitations": lim, "metrics": h})
    if cancelled and cancelled():
        return {"ok": False, "cancelled": True, "hypotheses": [], "best": None}
    hypotheses.extend(ldpc_hypotheses(bits, max_bits=max_bits))
    good_rs = [h for h in hypotheses if h["family"] == "reed-solomon" and h["confidence"] >= 0.25]
    good_conv = [h for h in hypotheses if h["family"] == "convolutional" and h["confidence"] >= 0.25]
    if good_rs and good_conv:
        best_rs, best_conv = good_rs[0], good_conv[0]
        conf = clamp01(min(best_rs["confidence"], best_conv["confidence"]) * 0.9)
        hypotheses.insert(0, {
            "family": "concatenated",
            "hypothesis": f"concatenated {best_conv['hypothesis']} + {best_rs['hypothesis']}",
            "confidence": float(conf), "status": "ok" if conf >= 0.3 else "low_confidence",
            "evidence": ["an outer Reed-Solomon signature and an inner convolutional-code signature "
                         "each passed their independent evidence tests",
                         f"inner: {best_conv['evidence'][0]}",
                         f"outer: {best_rs['evidence'][0]}"],
            "limitations": ["the layer order (inner/outer) is assumed from typical DSN/CCSDS "
                            "practice and is not proven by the data"],
            "components": [best_conv.get("code"), best_rs.get("code")]})
    hypotheses.sort(key=lambda h: -h["confidence"])
    best = hypotheses[0] if hypotheses else None
    if best is None or best["confidence"] < 0.2:
        # Evidence rule: a leader below the reporting threshold is *not* reported as a result.  The
        # full list is still returned so a user can inspect the near misses, but "best" stays None so
        # no downstream consumer (UI, report, comparison) can present a weak hypothesis as a finding.
        if best is not None:
            notes.append(f"the strongest hypothesis ({best['hypothesis']}, confidence "
                         f"{best['confidence']:.2f}) is below the 0.20 reporting threshold and is "
                         "therefore not claimed; the full candidate list is available for inspection")
        else:
            notes.append("no FEC hypothesis could be formed from this bitstream")
        best = None
        notes.append("no FEC family produced evidence strong enough to be claimed; the platform "
                     "reports 'no FEC identified' instead of guessing")
    else:
        notes.append(f"strongest FEC hypothesis: {best['hypothesis']} (confidence "
                     f"{best['confidence']:.2f}) - a hypothesis backed by the listed evidence, not "
                     "a verified fact")
    return {"ok": True, "hypotheses": hypotheses, "best": best,
            "n_bits_tested": int(min(bits.size, max_bits)), "notes": notes,
            "method": "hypothesis testing: hard-decision Viterbi distance vs a random-data null, "
                      "GF(256) syndrome/Berlekamp-Massey validation, and a rank statistic for "
                      "linear structure"}


# ---------------------------------------------------------------------------------------
# convenience record for the parameters table
# ---------------------------------------------------------------------------------------
def fec_param(result: dict) -> dict:
    """Summarise an :func:`analyse_fec` result as a parameter record."""
    best = (result or {}).get("best")
    if not best or best.get("confidence", 0.0) < 0.2:
        return unavailable("Forward-error-correction", "FEC", "no FEC family passed its evidence "
                           "test on the demodulated bits (see the FEC module for the tested "
                           "hypotheses and their scores)")
    return param("Forward-error-correction", best["hypothesis"], None, best["confidence"],
                 "best-scoring FEC hypothesis (Viterbi distance vs null / GF(256) syndrome "
                 "validation)", status=best.get("status", "ok"),
                 evidence=best.get("evidence", [])[:4], limitations=best.get("limitations", [])[:3])
