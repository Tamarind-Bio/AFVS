#!/usr/bin/env python3
"""Equivalence guard for predict_on_X(..., rows=...).

There is no pytest surface in this repo, so this is a runnable script rather than a test module.

WHAT IT GUARDS. `predict_on_X(model, meta, X, rows=idx)` must return exactly what the old spelling
`predict_on_X(model, meta, X[idx])` returned. The new path gathers per chunk and sorts within each
chunk for memory-mapped locality, inverting the permutation on the way out. An error in that
inversion does not crash and does not look wrong: it returns plausible scores attached to the WRONG
molecules, which silently permutes an acquisition ranking. That failure mode is the reason this
file exists.

Every assertion is paired with a control that must FAIL. A test that cannot fail certifies nothing,
and this one compares two code paths that could in principle be broken identically.

    python tools/templates/verify_predict_rows.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_regressor import (  # noqa: E402
    DEFAULT_CHUNK_BYTES,
    packed_width,
    predict_ensemble_on_X,
    predict_on_X,
)

FAILED = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def _model(n_bits, seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(n_bits, 16), nn.ReLU(), nn.Linear(16, 1))
    m.eval()
    return m, {"input_dim": n_bits, "fp_nbits": n_bits, "y_mean": -9.0, "y_std": 1.5}


def main():
    n_bits, N = 256, 8_000
    rng = np.random.RandomState(0)
    model, meta = _model(n_bits)

    Xf = rng.randint(0, 2, size=(N, n_bits)).astype(np.float32)
    Xp = np.packbits(Xf.astype(np.uint8), axis=1)
    assert Xp.shape[1] == packed_width(n_bits)

    print(f"chunk default: {DEFAULT_CHUNK_BYTES >> 20} MB unpacked "
          f"({DEFAULT_CHUNK_BYTES // (n_bits * 4):,} rows at {n_bits} bits)")

    print("\n=== rows= must equal the materialized gather ===")
    cases = {
        "sorted":     np.sort(rng.choice(N, 2000, replace=False)),
        "unsorted":   rng.choice(N, 2000, replace=False),   # the normal case: manifest order
        "duplicates": rng.choice(N, 2000, replace=True),    # two ligands can share a SMILES
        "single":     np.array([7]),
        "empty":      np.array([], dtype=np.int64),
    }
    for label, X in (("float32", Xf), ("packed", Xp)):
        for cname, idx in cases.items():
            old = predict_on_X(model, meta, X[idx])
            new = predict_on_X(model, meta, X, rows=idx)
            check(f"{label}/{cname}", old.shape == new.shape and np.array_equal(old, new),
                  f"n={len(idx)}")

    idx = rng.choice(N, 3000, replace=True)
    check("packed/many-chunks",
          np.array_equal(predict_on_X(model, meta, Xp[idx]),
                         predict_on_X(model, meta, Xp, rows=idx, chunk=97)))

    handles = [_model(n_bits, seed=s) for s in (1, 2, 3)]
    idx = rng.choice(N, 1500, replace=True)
    mo, vo = predict_ensemble_on_X(handles, Xp[idx])
    mn, vn = predict_ensemble_on_X(handles, Xp, rows=idx, chunk=257)
    check("ensemble/mean", np.array_equal(mo, mn))
    check("ensemble/var", np.array_equal(vo, vn))

    print("\n=== chunk size must not change the RESULT ===")
    idx = rng.choice(N, 2500, replace=False)
    ref = predict_on_X(model, meta, Xp, rows=idx, chunk=None)
    for ch in (17, 256, 4096, 10 * N):
        check(f"chunk={ch}", np.array_equal(ref, predict_on_X(model, meta, Xp, rows=idx, chunk=ch)))

    print("\n=== controls: these prove the comparison above can fail ===")
    good = predict_on_X(model, meta, Xp, rows=idx)
    check("reversed rows differ", not np.array_equal(good, predict_on_X(model, meta, Xp, rows=idx[::-1])))
    bad = idx.copy(); bad[0] = (bad[0] + 1) % N
    check("one changed index differs", not np.array_equal(good, predict_on_X(model, meta, Xp, rows=bad)))
    try:
        predict_on_X(model, meta, Xp, rows=np.zeros((3, 3), dtype=np.int64))
        check("2-D rows raises", False, "no exception")
    except ValueError:
        check("2-D rows raises", True)

    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): {FAILED}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
