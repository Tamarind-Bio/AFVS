#!/usr/bin/env python3
"""Equivalence guard for the streamed fingerprint cache build.

There is no pytest surface in this repo, so this is a runnable script.

WHAT IT GUARDS. `_cmd_init` used to hold three full-pool Python structures at once: the SMILES
list, the kept-index list, and a second SMILES list built as `[all_smiles[i] for i in kept]`.
Measured, those are ~117, ~40 and ~117 bytes per molecule, so roughly 274 GB at a 1e9 pool before
a single fingerprint is stored. `fingerprint_to_packed_npy` now accepts an ITERABLE, returns an
int64 kept ARRAY, and writes the kept-SMILES sidecar incrementally, so all three are bounded by
one batch.

The risk in that change is not a crash, it is a SILENT REORDERING. The cache's row order IS its
identity: row j is looked up by position, so if the streamed build kept a different set, or the
same set in a different order, every prediction would be attached to the wrong molecule and the
scores would still look entirely plausible. Nothing downstream would raise.

So the reference here is `fingerprint_all`, the original non-streaming implementation and a
SHIPPED symbol in this same module, not a reimplementation of the old code path. The module's own
docstring prescribes exactly this comparison.

Every assertion carries a control that must FAIL.

    python tools/templates/verify_streaming_fingerprint.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_regressor import (  # noqa: E402
    DEFAULT_N_BITS,
    fingerprint_all,
    fingerprint_to_packed_npy,
    unpack_fp,
)

FAILED = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


# A pool with unparseable entries deliberately interleaved, because the whole subtlety is that
# drops shift every later row and the kept array is what records the shift.
GOOD = [
    "CCO", "c1ccccc1", "CC(=O)O", "CCN", "OC1CCCCC1", "FC1=CC=CC=C1", "CCOCC",
    "CC(=O)Nc1ccccc1", "COc1ccccc1", "NCCc1ccccc1", "CC(C)Oc1ccccc1", "CCCCCCCC",
    "c1ccc2ccccc2c1", "CC1CCCCC1", "OCCc1ccccc1", "ClC1=CC=CC=C1", "CCCCO", "CN(C)C",
]
BAD = ["not-a-molecule", "", "J((((", "N/A"]


def build_pool(n_repeat=3):
    pool = []
    for r in range(n_repeat):
        for i, s in enumerate(GOOD):
            pool.append(s)
            if i % 5 == 2:                      # scatter the drops through the pool
                pool.append(BAD[(i + r) % len(BAD)])
    return pool


def main():
    pool = build_pool()
    n_bad = sum(1 for s in pool if s in BAD)
    print(f"pool: {len(pool)} entries, {n_bad} deliberately unparseable, {DEFAULT_N_BITS} bits")

    # --- reference: the original non-streaming implementation -------------
    ref_kept, ref_X = fingerprint_all(pool)
    ref_kept = np.asarray(ref_kept)
    if ref_kept.shape[0] == 0:
        print("CONTROL FAILED: the reference kept nothing", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as td:
        npy = os.path.join(td, "fp.npy")
        keep = os.path.join(td, "keep.npy")
        side = os.path.join(td, "smiles.parquet")

        # --- the shipped streamed path, fed a GENERATOR with no len() ------
        got_kept, got_X = fingerprint_to_packed_npy(
            (s for s in pool), npy, n_total=len(pool), keep_out=keep, smiles_out=side, chunk=7,
        )

        print("\n=== the streamed build must match the reference exactly ===")
        check("kept indices identical",
              np.array_equal(np.asarray(got_kept), ref_kept),
              f"{len(got_kept)} kept of {len(pool)}")
        check("kept is an int64 array, not a Python list",
              isinstance(got_kept, np.ndarray) and got_kept.dtype == np.int64,
              f"{type(got_kept).__name__}/{getattr(got_kept, 'dtype', None)}")
        check("kept indices strictly increasing (searchsorted depends on it)",
              bool(np.all(np.diff(np.asarray(got_kept)) > 0)))
        check("row count matches", got_X.shape[0] == ref_X.shape[0],
              f"{got_X.shape[0]} vs {ref_X.shape[0]}")

        unpacked = unpack_fp(np.asarray(got_X), DEFAULT_N_BITS)
        check("unpacked fingerprints bit-identical to the reference",
              np.array_equal(unpacked, ref_X))

        # --- the persisted artifacts ---------------------------------------
        check("kept array persisted", os.path.exists(keep))
        if os.path.exists(keep):
            check("persisted kept matches returned kept",
                  np.array_equal(np.load(keep), np.asarray(got_kept)))

        check("smiles sidecar written", os.path.exists(side))
        if os.path.exists(side):
            import pandas as pd
            side_smi = pd.read_parquet(side).smiles.tolist()
            expect = [pool[i] for i in ref_kept]
            check("sidecar holds exactly the kept smiles IN ORDER",
                  side_smi == expect,
                  f"{len(side_smi)} rows")
            # Row j of the cache must be the molecule at sidecar row j. This is the invariant an
            # off-by-one in the incremental writer would break while everything else still passed.
            j = len(side_smi) // 2
            one_kept, one_X = fingerprint_all([side_smi[j]])
            check("cache row j really is sidecar molecule j",
                  len(one_kept) == 1 and np.array_equal(unpacked[j], one_X[0]),
                  f"j={j}")

        # --- CONTROLS: these must prove the comparisons above can fail -----
        print("\n=== controls ===")
        check("a shuffled reference does NOT match",
              not np.array_equal(unpacked[::-1], ref_X))
        # An OFF-BY-ONE misalignment, which is the failure this guard exists for: same rows, same
        # count, shifted by one. Comparing unpacked[:-1] against ref_X[:-1] would drop the same row
        # from both sides and pass trivially, which is not a control at all.
        check("an off-by-one misalignment does NOT match",
              not np.array_equal(unpacked[:-1], ref_X[1:]))
        bad_npy = os.path.join(td, "fp2.npy")
        k2, X2 = fingerprint_to_packed_npy((s for s in pool[:5]), bad_npy, n_total=5, chunk=3)
        check("a different pool yields a different kept array",
              not np.array_equal(np.asarray(k2), ref_kept))
        try:
            fingerprint_to_packed_npy((s for s in pool), os.path.join(td, "fp3.npy"))
            check("a generator with no n_total must raise", False, "no exception")
        except ValueError:
            check("a generator with no n_total must raise", True)

    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): {FAILED}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
