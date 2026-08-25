#!/usr/bin/env python3
"""Guard for the fingerprint cache's COVERAGE floor in afvs_al_controller._load_fp_cache.

There is no pytest surface in this repo, so this is a runnable script.

WHAT IT GUARDS. `_assert_cache_consistent` compares the cache's artifacts against EACH OTHER, which
catches a torn init. It cannot answer the different question of whether the cache covers the whole
MANIFEST, because a cache that is merely SHORT satisfies every internal check: fp_keep, the SMILES
sidecar and the .npy all agree and all stop early. That state is reachable without any exotic
failure, since `fingerprint_to_packed_npy` flushes the sidecar per chunk and saves fp_keep once at
the end, so an init killed partway leaves exactly it.

The consequence is silent and is the failure this whole design exists to prevent: rows absent from
the cache resolve to row -1, score +inf and are never selected, so the campaign screens a FRACTION
of the library while every stage reports success and the round returns CONTINUE.

Coverage cannot be inferred from the artifacts, and that is the crux. A cache missing rows because
the init never reached them and one missing rows because those molecules were UNPARSEABLE are the
same shape on disk. Only the row count the init ran over separates them, so `_cmd_init` records it
in fp_cache_meta.json after every other artifact is written, and `_assert_cache_covers_manifest`
requires it.

The load-bearing case here is CASE 2, not the rejections. A check that refused every cache shorter
than its manifest would pass all the rejection cases and break every real screen, because a
sub-1-percent parse-failure rate is normal. Case 2 is what separates the fix from that.

    python tools/verify_cache_coverage.py
"""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "templates"))

import afvs_al_controller as C  # noqa: E402

N_BITS = 2048
WIDTH = N_BITS // 8
FAILURES = []


def check(ok, label, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        FAILURES.append(label)


def build(tmp, n_manifest, kept_rows, write_keep=True, sidecar_rows=None, meta_total="match",
          legacy_npz=False):
    """Materialise a state dir.

    `kept_rows` is the explicit list of MANIFEST ROWS the cache holds, so a fixture can express
    "covers a prefix" and "covers everything except a few unparseable rows" as different shapes
    rather than as one length. `meta_total`: "match" writes n_manifest, None writes no file at all
    (a killed init), an int writes that value.

    SMILES are distinct per row. That matters for the legacy fallback, which matches the cache's
    smiles against the manifest's by VALUE, so a repeated string would collapse every manifest row
    onto one cache row and make the result an artifact of the fixture.
    """
    os.makedirs(tmp, exist_ok=True)
    st = C._state(tmp)
    smis = [f"C{'C' * (i % 7)}O{i}" for i in range(n_manifest)]
    pd.DataFrame({"collection_key": [f"AA_{i // 5:07d}" for i in range(n_manifest)],
                  "ligand_id": [f"L{i}" for i in range(n_manifest)],
                  "smiles": smis}).to_parquet(st["manifest"], index=False)
    kept = np.asarray(kept_rows, dtype=np.int64)
    X = np.full((max(len(kept), 1), WIDTH), 7, dtype=np.uint8)
    cache_smis = [smis[i] for i in kept]
    if legacy_npz:
        np.savez(st["fp"], X=X, smiles=np.array(cache_smis, dtype=object))
    else:
        np.save(st["fp_npy"], X)
        rows = len(kept) if sidecar_rows is None else sidecar_rows
        pq.write_table(pa.table({"smiles": pa.array(cache_smis[:rows], type=pa.string())}),
                       st["fp_smiles"])
        if write_keep:
            np.save(st["fp_keep"], kept)
    if meta_total is not None:
        total = n_manifest if meta_total == "match" else meta_total
        with open(st["fp_meta"], "w") as fh:
            json.dump({"n_total": int(total), "n_kept": int(len(kept))}, fh)
    return st


def probe(st, n_manifest):
    """(verdict, detail). ACCEPTED means _load_fp_cache handed back a usable cache.

    `detail` is the FULL message, truncated only at print time. Truncating here made the
    message-quality asserts below fail against their own elision rather than against the code.
    """
    try:
        row_of, _X = C._load_fp_cache(st, n_manifest)
    except Exception as e:
        return "REJECTED", f"{type(e).__name__}: {' '.join(str(e).split())}"
    resolved = int((np.asarray(row_of) >= 0).sum())
    return "ACCEPTED", f"resolves {resolved:,}/{n_manifest:,} ({100.0 * resolved / n_manifest:.1f}%)"


N = 1000
print("\nfingerprint cache coverage floor")
with tempfile.TemporaryDirectory() as tmp:
    ALL = list(range(N))
    PREFIX = list(range(300))                       # an init killed after 300 rows
    SPARSE = [i for i in ALL if i % 97]             # every row EXCEPT ~1% unparseable, spread out

    cases = [
        # (key, expected, label, state)
        ("1", "ACCEPTED", "CONTROL full cache with a coverage record is accepted",
         build(os.path.join(tmp, "c1"), N, ALL)),
        ("2", "ACCEPTED", "CONTROL cache missing ~1% UNPARSEABLE rows is still accepted",
         build(os.path.join(tmp, "c2"), N, SPARSE)),
        ("3", "REJECTED", "killed init: short cache, no coverage record",
         build(os.path.join(tmp, "c3"), N, PREFIX, meta_total=None)),
        ("4", "REJECTED", "killed init on the legacy sidecar branch (fp_keep absent)",
         build(os.path.join(tmp, "c4"), N, PREFIX, write_keep=False, meta_total=None)),
        ("5", "REJECTED", "legacy .npz branch with no coverage record",
         build(os.path.join(tmp, "c5"), N, PREFIX, meta_total=None, legacy_npz=True)),
        ("6", "REJECTED", "cache built against a DIFFERENT manifest (record disagrees)",
         build(os.path.join(tmp, "c6"), N, ALL, meta_total=N // 2)),
        ("7", "REJECTED", "CONTROL the pre-existing torn-pair check still fires",
         build(os.path.join(tmp, "c7"), N, list(range(200)), sidecar_rows=100)),
    ]
    details = {}
    for key, expected, label, st in cases:
        verdict, detail = probe(st, N)
        details[key] = detail
        check(verdict == expected, f"{expected:<9} {label}", detail[:78])

    # The rejection for a MISSING record must name the remedy. Without this assert, degrading that
    # branch to a bare FileNotFoundError still refuses the cache and so still passes every case
    # above, while handing the operator no idea what to do; the explicit branch exists for exactly
    # that message.
    for key in ("3", "4", "5"):
        d = details.get(key, "")
        check("coverage record" in d and "re-run init" in d,
              f"case {key}: refusal names the cause AND the remedy", d[:70])

    # Case 2 is the one that separates this fix from "reject anything short", so assert the fixture
    # really is short. Without this the control passes vacuously on a full cache.
    n_sparse = len(SPARSE)
    check(n_sparse < N, "CONTROL case 2's cache really is shorter than its manifest",
          f"{n_sparse:,} kept of {N:,}")
    # And that the ACCEPTED verdict in case 2 genuinely left those rows unresolved, rather than the
    # loader quietly resolving them some other way, which would make the case prove nothing.
    row_of, _ = C._load_fp_cache(build(os.path.join(tmp, "c2b"), N, SPARSE), N)
    unresolved = int((np.asarray(row_of) < 0).sum())
    check(unresolved == N - n_sparse,
          "CONTROL case 2's uncached rows really do resolve to -1",
          f"{unresolved} unresolved == {N - n_sparse} uncached")

    # END TO END, and this is the case the hand-built fixtures above structurally cannot cover: they
    # write the coverage record THEMSELVES, so every one of them passes even if the shipped writer
    # never writes it at all. The reader and the writer are separate halves of one contract, and only
    # driving the real _cmd_init tests the link. Uses the collection-ligands seam so nothing touches S3.
    import argparse                                                             # noqa: E402
    e2e = os.path.join(tmp, "e2e")
    os.makedirs(e2e, exist_ok=True)
    real_smis = ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "C1CCCCC1", "CCOCC", "CCCCO", "c1ccncc1"]
    cl_path = os.path.join(e2e, "cl.parquet")
    store_path = os.path.join(e2e, "store.parquet")
    pd.DataFrame({"collection_key": [f"AA_{i // 3:07d}" for i in range(len(real_smis))],
                  "ligand_id": [f"L{i}" for i in range(len(real_smis))]}).to_parquet(cl_path)
    pd.DataFrame({"ligand_id": [f"L{i}" for i in range(len(real_smis))],
                  "smiles": real_smis}).to_parquet(store_path)
    sd = os.path.join(e2e, "state")
    C._cmd_init(argparse.Namespace(
        state_dir=sd, out=os.path.join(e2e, "todo.all"), collection_ligands=cl_path,
        collection_prefix="", data_bucket="", todo="", smiles_store=store_path,
        seed_frac=0.5, min_seed=2, budget=0, seed=42, n_workers=1))
    st_e2e = C._state(sd)
    check(os.path.exists(st_e2e["fp_meta"]),
          "the SHIPPED _cmd_init writes the coverage record", st_e2e["fp_meta"])
    n_real = C.parquet_num_rows(st_e2e["manifest"])
    v, d = probe(st_e2e, n_real)
    check(v == "ACCEPTED", "a cache written by the real init is accepted by the real reader", d)

print("\n" + "=" * 76)
if FAILURES:
    print(f"FAILED ({len(FAILURES)}):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("ALL CHECKS PASSED: a cache that does not cover its manifest is refused on every read path,")
print("including the legacy sidecar and .npz ones, while a cache legitimately missing unparseable")
print("rows is still accepted.")
