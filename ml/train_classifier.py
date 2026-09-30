#!/usr/bin/env python3
"""Train the classical-ML tie-breaker for automatic modulation classification.

Generates a labelled dataset with the platform's own synthetic generator (the same
code path used for the demo data and the benchmark), extracts the DSP feature vector
used by the hybrid classifier and fits a scikit-learn RandomForest.  The held-out
accuracy and confusion matrix are printed and stored next to the model so the app can
report exactly how good the tie-breaker is - it is never presented as ground truth.

Usage:
    python ml/train_classifier.py --n-per-class 180 --snr-min 4 --snr-max 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsp import modulation as md          # noqa: E402
from dsp import params as pm              # noqa: E402
from dsp import synth                     # noqa: E402
from dsp import spectrum as sp            # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "modclass.joblib")

MODS = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "GFSK", "2FSK", "AM", "FM"]
# The FSK variants are kept as separate classes: CPFSK with a rectangular frequency pulse (2FSK)
# and Gaussian-smoothed CPFSK (GFSK) are distinguishable through the IF edge-sharpness feature.
LABEL_MAP: dict[str, str] = {}


def make_sample(rng: np.random.Generator, mod: str, snr: float) -> tuple[np.ndarray, float, dict]:
    fs = float(rng.choice([100e3, 200e3, 250e3, 400e3]))
    rs = float(rng.choice([5e3, 10e3, 12.5e3, 20e3, 25e3, 40e3, 50e3]))
    spec = {
        "modulation": mod, "fs": fs, "symbol_rate": rs, "snr_db": float(snr),
        "carrier_offset_hz": float(rng.uniform(-fs / 8, fs / 8)) if mod not in ("AM", "FM") else 0.0,
        "rolloff": float(rng.choice([0.2, 0.25, 0.35, 0.5])),
        "n_symbols": int(rng.integers(1500, 4000)),
        "mod_index": float(rng.choice([0.5, 0.7, 1.0])),
        "bt": float(rng.choice([0.3, 0.35, 0.5])),
        "fm_deviation_hz": float(rng.choice([5000.0, 15000.0, 25000.0])),
        "carrier_hz": float(fs * rng.choice([0.15, 0.2, 0.25])) if mod in ("AM", "FM") else None,
        "seed": int(rng.integers(0, 1 << 30)),
    }
    for k in ("carrier_hz",):
        if spec.get(k) is None:
            spec.pop(k)
    g = synth.generate_signal(spec)
    return g.samples, g.fs, g.ground_truth


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=160)
    ap.add_argument("--snr-min", type=float, default=4.0)
    ap.add_argument("--snr-max", type=float, default=30.0)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=MODEL_PATH)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    X, y = [], []
    t0 = time.time()
    for mod in MODS:
        made = 0
        attempts = 0
        while made < args.n_per_class and attempts < args.n_per_class * 4:
            attempts += 1
            snr = float(rng.uniform(args.snr_min, args.snr_max))
            try:
                x, fs, gt = make_sample(rng, mod, snr)
                a = sp.analyse_spectrum(x, fs, nperseg=int(min(4096, max(256, x.size // 8))))
                if not a.get("ok"):
                    continue
                rs_est = gt.get("symbol_rate") if mod not in ("AM", "FM") else None
                fe = md.extract_features(x, fs, rs_est, a.get("obw_99_hz"))
                if not fe.get("ok"):
                    continue
                vec = [fe["features"].get(k) for k in md.FEATURE_NAMES]
                vec = [0.0 if v is None or not np.isfinite(v) else float(v) for v in vec]
                X.append(vec)
                y.append(LABEL_MAP.get(mod, mod))
                made += 1
            except Exception as exc:                     # keep the training run alive
                print(f"  ! {mod} sample failed: {exc}", file=sys.stderr)
        print(f"{mod:6s}: {made} vectors  ({time.time()-t0:.0f}s elapsed)")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if X.size == 0:
        print("no samples generated", file=sys.stderr)
        return 1

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    import joblib

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=args.seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=args.trees, random_state=args.seed,
                                 class_weight="balanced_subsample", n_jobs=-1, min_samples_leaf=2)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = float(accuracy_score(yte, pred))
    classes = [str(c) for c in clf.classes_]
    cm = confusion_matrix(yte, pred, labels=clf.classes_).tolist()
    print(f"\nheld-out accuracy: {acc:.3f} on {Xte.shape[0]} vectors")
    print("confusion matrix (rows = truth, cols = prediction)")
    print("        " + " ".join(f"{c:>6s}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:>7s} " + " ".join(f"{v:6d}" for v in cm[i]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({"model": clf, "feature_names": md.FEATURE_NAMES, "classes": classes,
                 "accuracy": acc, "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
                 "confusion_matrix": cm, "snr_range": [args.snr_min, args.snr_max],
                 "name": "random-forest-amc", "modulations": MODS,
                 "label_map": LABEL_MAP}, args.out)
    with open(os.path.join(os.path.dirname(args.out), "modclass_report.json"), "w") as fh:
        json.dump({"accuracy": acc, "classes": classes, "confusion_matrix": cm,
                   "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
                   "feature_names": md.FEATURE_NAMES, "snr_range": [args.snr_min, args.snr_max],
                   "modulations": MODS, "trained_s": round(time.time() - t0, 1)}, fh, indent=2)
    print(f"\nmodel written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
