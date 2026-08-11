#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Active-learning acquisition step for multi-round virtual screening (TAM-1599).

The regress-and-rank analogue of create_todofile_atg-primaryscreen.py: instead of scoring tranches
by the heuristic min/avg prescreen score, score every collection by a TRAINED surrogate's predicted
best score over its ligands, then greedily select the best collections up to a ligand budget and
write the next todo.all. This is the acquisition unit of the AL loop; the controller
(afvs_al_controller.py) calls it each round with the surrogate trained on everything docked so far.

Granularity is per-collection (the AFVS atomic docking + S3-fetch unit), which is chemically
meaningful because a tranche's collections are property-homogeneous. It is non-lossy: a collection
passed over early can be selected later once the improved surrogate ranks it up (unlike a monotonic
classify-and-prune).

The (collection -> ligands -> SMILES) MANIFEST is an INPUT, not built here, because its cheapest
source is deployment-specific (collection tarball file lists at COLLECTION_PREFIX, a stored
ingestion manifest, or the ligands store joined to a ligand->collection map). afvs_al_controller.py
builds it; keeping it an input makes this step pure + unit-testable and lets the manifest source
change without touching the acquisition logic.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"))
from ml_regressor import fingerprint_all, predict_on_X, load_regressor  # noqa: E402


def per_collection_scores(model, meta, manifest, fp_cache=None):
    """
    Predict a per-collection acquisition score = the BEST (most negative) predicted ligand score in
    the collection (mirrors ATG tranche_min_score, at collection granularity).

    Args:
        model, meta: trained surrogate (ml_regressor).
        manifest (DataFrame): columns collection_key, ligand_id, smiles.
        fp_cache (dict smiles->row-index, np.ndarray X) optional precomputed fingerprint cache.

    Returns:
        DataFrame: collection_key, pred_best, ligand_count.
    """
    smiles = manifest["smiles"].astype(str).tolist()
    if fp_cache is not None:
        smi_to_row, X = fp_cache
        rows = np.array([smi_to_row.get(s, -1) for s in smiles])
        pred = np.full(len(smiles), np.nan, dtype=np.float32)
        ok = rows >= 0
        if ok.any():
            pred[ok] = predict_on_X(model, meta, X[rows[ok]])
    else:
        kept, X = fingerprint_all(smiles, radius=meta.get("fp_radius", 2), n_bits=meta.get("fp_nbits", 1024))
        pred = np.full(len(smiles), np.nan, dtype=np.float32)
        if len(kept):
            p = predict_on_X(model, meta, X)
            for li, oi in enumerate(kept):
                pred[oi] = p[li]

    df = manifest.copy()
    df["pred"] = pred
    # unscoreable ligands (bad SMILES) sink to the worst score so they don't inflate a collection
    df["pred"] = df["pred"].fillna(np.inf)
    agg = df.groupby("collection_key").agg(pred_best=("pred", "min"),
                                           ligand_count=("ligand_id", "size")).reset_index()
    return agg


def select_collections(agg, docked_collections, budget_ligands, epsilon_random=0.0, seed=42):
    """
    Greedily select the best-predicted un-docked collections up to a ligand budget, plus an optional
    epsilon-random exploration fraction. Returns the list of selected collection_keys.
    """
    avail = agg[~agg.collection_key.isin(set(docked_collections))].copy()
    if avail.empty:
        return []

    rng = np.random.RandomState(seed)
    n_random = int(round(budget_ligands * epsilon_random))
    selected, total = [], 0

    # exploration: a few random collections first (hedge at collection granularity)
    if n_random > 0:
        rand_order = rng.permutation(len(avail))
        for idx in rand_order:
            row = avail.iloc[idx]
            selected.append(row.collection_key)
            total += int(row.ligand_count)
            if total >= n_random:
                break
        avail = avail[~avail.collection_key.isin(set(selected))]

    # exploitation: greedy by predicted best score (ascending = best binder first)
    for _, row in avail.sort_values("pred_best").iterrows():
        if total >= budget_ligands:
            break
        selected.append(row.collection_key)
        total += int(row.ligand_count)
    return selected


def write_todo(agg, selected, out_path):
    """Write the next todo.all: `TrancheCollection LigandCount` lines for the selected collections."""
    sel = agg[agg.collection_key.isin(set(selected))][["collection_key", "ligand_count"]]
    sel.to_csv(out_path, index=False, header=False, sep=" ")
    return len(sel), int(sel.ligand_count.sum())


def predict_ligand_scores(model, meta, manifest, fp_cache=None):
    """
    Predict a per-LIGAND docking score for every row of the manifest (columns collection_key,
    ligand_id, smiles). Returns the manifest with a 'pred' column (unscoreable ligands -> +inf).
    This is the per-molecule path: rank individual ligands, not collections (recall 0.75 vs the
    collection-granular 0.14 on real Tamarind qvina02 runs, see notes/collection-granularity-finding.md).
    """
    smiles = manifest["smiles"].astype(str).tolist()
    if fp_cache is not None:
        smi_to_row, X = fp_cache
        rows = np.array([smi_to_row.get(s, -1) for s in smiles])
        pred = np.full(len(smiles), np.nan, dtype=np.float32)
        ok = rows >= 0
        if ok.any():
            pred[ok] = predict_on_X(model, meta, X[rows[ok]])
    else:
        kept, X = fingerprint_all(smiles, radius=meta.get("fp_radius", 2), n_bits=meta.get("fp_nbits", 1024))
        pred = np.full(len(smiles), np.nan, dtype=np.float32)
        if len(kept):
            p = predict_on_X(model, meta, X)
            for li, oi in enumerate(kept):
                pred[oi] = p[li]
    df = manifest.copy()
    df["pred"] = np.where(np.isnan(pred), np.inf, pred)
    return df


def select_ligands(manifest_pred, docked_ligands, budget_ligands, epsilon_random=0.05, seed=42):
    """
    Rank un-docked ligands by predicted score (ascending = best binder first), take the top
    `budget_ligands` (with an optional epsilon-random exploration fraction). Returns the selected
    sub-DataFrame (collection_key, ligand_id, smiles, pred).
    """
    # coerce both sides to str: manifest ligand_id can be numeric (e.g. from a pre-built --collection-ligands
    # parquet) while the exclusion set is str, and a dtype mismatch would silently re-select docked ligands.
    # reset_index so selection is POSITIONAL: a non-unique manifest index (a future manifest source, a concat)
    # would make label-based `.loc[picked_idx]` fan out (one label -> many rows), silently over-selecting and
    # blowing the docking budget. Positional row ids are unique by construction, so this can't over-select.
    avail = manifest_pred[~manifest_pred.ligand_id.astype(str).isin({str(x) for x in docked_ligands})] \
        .reset_index(drop=True)
    if avail.empty or budget_ligands <= 0:
        return avail.iloc[0:0]
    rng = np.random.RandomState(seed)
    n_random = min(len(avail), int(round(budget_ligands * epsilon_random)))
    picked_pos = []
    if n_random > 0:
        picked_pos = list(rng.choice(avail.index.to_numpy(), n_random, replace=False))
    rest = avail.drop(index=picked_pos).sort_values("pred")
    n_greedy = max(0, budget_ligands - len(picked_pos))
    picked_pos += list(rest.index[:n_greedy])
    return avail.loc[picked_pos]


def write_named_csv(selected, out_path):
    """
    Emit an AFVS `csv_collection_key_ligand` collection list for the selected ligands: header
    `collection_name,collection_number,ligand`, one row per ligand. afvs_prepare_folders.py parses
    this into per-collection `named`-mode entries, so AFVS docks ONLY these ligands (afvs_run.py
    unpack_item named-mode). collection_key = `<name>_<number>`.
    """
    parts = selected.collection_key.astype(str).str.rsplit("_", n=1, expand=True)
    out = pd.DataFrame({"collection_name": parts[0], "collection_number": parts[1],
                        "ligand": selected.ligand_id.astype(str)})
    out.to_csv(out_path, index=False)
    return len(out)


def acquire(model_path, manifest_path, docked_collections, budget_ligands, out_path,
            epsilon_random=0.0, seed=42, fp_cache=None):
    model, meta = load_regressor(model_path)
    manifest = pd.read_parquet(manifest_path) if manifest_path.endswith(".parquet") \
        else pd.read_csv(manifest_path)
    agg = per_collection_scores(model, meta, manifest, fp_cache=fp_cache)
    selected = select_collections(agg, docked_collections, budget_ligands, epsilon_random, seed)
    n_coll, n_lig = write_todo(agg, selected, out_path)
    print(f"selected {n_coll} collections / {n_lig} ligands -> {out_path}")
    return selected


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--selftest":
        ap = argparse.ArgumentParser(description=__doc__)
        ap.add_argument("--model", required=True, help="trained surrogate .pt (ml_regressor)")
        ap.add_argument("--manifest", required=True, help="parquet/csv: collection_key, ligand_id, smiles")
        ap.add_argument("--docked", default="", help="comma/file list of already-docked collection_keys")
        ap.add_argument("--budget", type=int, required=True, help="ligand budget for this round")
        ap.add_argument("--out", required=True, help="output todo.all path")
        ap.add_argument("--epsilon-random", type=float, default=0.0)
        ap.add_argument("--seed", type=int, default=42)
        args = ap.parse_args()
        docked = []
        if args.docked:
            docked = open(args.docked).read().split() if os.path.isfile(args.docked) else args.docked.split(",")
        acquire(args.model, args.manifest, docked, args.budget, args.out, args.epsilon_random, args.seed)
        sys.exit(0)

    # --- self-test: synthetic manifest with learnable signal, no docking needed ---
    import tempfile
    from rdkit.Chem import Descriptors
    from rdkit import Chem
    from ml_regressor import train_regressor, save_regressor

    demo = ['CCO', 'CCCO', 'CCCCO', 'CCN', 'CCCN', 'c1ccccc1', 'c1ccccc1C', 'c1ccccc1O', 'CC(=O)O',
            'CC(=O)N', 'c1ccncc1', 'C1CCCCC1', 'CC(C)C', 'CCCCCCCCCC', 'c1ccc2ccccc2c1',
            'NC(Cc1ccccc1)C(=O)O', 'OC1CCCCC1', 'FC1=CC=CC=C1', 'CCOCC', 'CCC(=O)CC',
            'c1ccc(cc1)C(=O)O', 'CC(=O)Nc1ccccc1', 'COc1ccccc1', 'NCCc1ccccc1', 'CC(C)Oc1ccccc1']
    rng = np.random.RandomState(0)
    # score correlates with heavy-atom count (learnable); more atoms = more negative (better)
    scores = np.array([-1.0 * Descriptors.HeavyAtomCount(Chem.MolFromSmiles(s)) + rng.normal(0, 1.0)
                       for s in demo], dtype=np.float32)

    # train the surrogate on a "docked" seed (first 15), then acquire over all collections
    model, meta = train_regressor(demo[:15], scores[:15], seed=0, verbose=False)
    with tempfile.TemporaryDirectory() as td:
        mp = os.path.join(td, "m.pt")
        save_regressor(model, meta, mp)
        # build a manifest: 5 collections of 5 ligands each; collection i holds demo[5i:5i+5]
        rows = []
        for c in range(5):
            for s in demo[5 * c:5 * c + 5]:
                rows.append({"collection_key": f"AA_{c:07d}", "ligand_id": s, "smiles": s})
        manifest = pd.DataFrame(rows)
        mfp = os.path.join(td, "manifest.parquet")
        manifest.to_parquet(mfp)
        out = os.path.join(td, "todo.all")
        # already docked = collection AA_0000000 (the seed's first collection); budget 10 ligands
        selected = acquire(mp, mfp, ["AA_0000000"], budget_ligands=10, out_path=out, seed=0)

        assert "AA_0000000" not in selected, "docked collection must not be re-selected"
        assert 0 < len(selected) <= 4, f"expected 1-4 collections, got {len(selected)}"
        # the highest-heavy-atom collection (AA_0000002 = decane + naphthalenes) should rank in
        agg = per_collection_scores(model, meta, manifest)
        best_coll = agg[agg.collection_key != "AA_0000000"].sort_values("pred_best").iloc[0].collection_key
        assert best_coll in selected, f"best-predicted collection {best_coll} not selected"
        lines = open(out).read().strip().split("\n")
        assert all(len(ln.split()) == 2 for ln in lines), "todo.all lines must be 'Collection Count'"
        print(f"self-test PASSED: selected {selected}, best-predicted {best_coll} included, "
              f"todo.all well-formed ({len(lines)} lines)")
