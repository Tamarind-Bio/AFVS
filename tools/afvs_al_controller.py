#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Active-learning controller for multi-round virtual screening (TAM-1599).

Runs on the AFVS head node. Drives the seed -> train -> predict -> acquire -> dock loop, reusing the
existing AFVS docking fan-out for each round (run-virtual-screening.sh submits the todo.all this
controller writes, then invokes the next `round`). Two subcommands:

  init   build the (collection -> ligand -> SMILES) manifest + fingerprint cache once, and write the
         SEED todo.all (a random subset of collections). State goes to <state-dir>.
  round  read the docking scores accumulated so far, train the regression surrogate on them, acquire
         the next batch of collections (afvs_al_select), write the next todo.all, and report whether
         to CONTINUE / CONVERGED / BUDGET. Called once per round by the run driver.

Design decisions (validated against exhaustively-docked ground truth):
  - regress-and-rank surrogate (ml_regressor), retrained from scratch each round (online was worse).
  - acquisition is MolPAL's (vendored molpal.acquirer), at MolPAL's own defaults: greedy,
    epsilon 0. The explore round uses MolPAL's `random` metric.
  - convergence: stop when the rolling mean of the top-K docked score stops improving, or the docking
    budget is spent, whichever first.
  - guardrail 2: after the seed round, if the held-out surrogate Spearman is below --min-spearman,
    print a QUALITY_LOW warning so the run driver can fall back to brute force.

TWO infra seams (marked SEAM below, cannot be unit-tested off a live run):
  (A) build_manifest: the cheapest (collection -> ligand_ids) source is deployment-specific (collection
      tarball file lists at COLLECTION_PREFIX, or a stored ingestion manifest). Confirm with the AFVS
      owner. The SMILES join (ligand_id -> smiles) is the ligands store parquet (validated).
  (B) load_docked_scores: reads the run's accumulated per-workunit parquets (ligand, collection_key,
      score_min), the same artifacts run-virtual-screening.sh converts to results.csv.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"))
from ml_regressor import (fingerprint_all, train_on_X, train_ensemble_on_X,  # noqa: E402
                          predict_on_X, save_regressor, load_regressor)
from afvs_al_select import (per_collection_scores, select_collections, write_todo,  # noqa: E402
                            predict_ligand_scores, predict_ligand_scores_ensemble,
                            select_ligands, write_named_csv, env_setting)

# MolPAL metrics that actually READ y_vars. greedy and random ignore it, so an ensemble would be
# pure cost for them. Both spellings of Thompson sampling are listed because MolPAL's get_metric
# maps "thompson" and "ts" to the same function: listing only one made the alias skip the ensemble
# and then run on all-zero variance, which is the silent-degradation this set exists to prevent.
VAR_METRICS = {"ucb", "lcb", "ei", "pi", "ts", "thompson"}

# What we are willing to run, which is deliberately NARROWER than MolPAL's get_metric table.
# Excluded on purpose: "threshold" takes a threshold= we never pass, so MolPAL's default -inf makes
# it return a constant, i.e. a fully RANDOM screen billed at full docking cost; and "noisy" wraps a
# base metric we never configure. Both are accepted by get_metric and neither would raise.
SUPPORTED_METRICS = {"greedy", "random"} | VAR_METRICS

try:
    from scipy.stats import spearmanr
except Exception:  # scipy optional; quality gate degrades to "unknown" without it
    spearmanr = None


# ---------- state ----------

def _state(state_dir):
    return {
        "manifest": os.path.join(state_dir, "manifest.parquet"),
        "fp": os.path.join(state_dir, "fp_cache.npz"),
        "docked": os.path.join(state_dir, "docked_collections.txt"),
        "model": os.path.join(state_dir, "model.pt"),
        "history": os.path.join(state_dir, "history.json"),
    }


def _load_fp_cache(path):
    d = np.load(path, allow_pickle=True)
    smi = list(d["smiles"])
    return {s: i for i, s in enumerate(smi)}, d["X"]


# ---------- SEAM A: manifest ----------

def build_manifest(collection_ligands, smiles_store_path, out_path):
    """
    Join (collection_key -> ligand_id) to (ligand_id -> smiles) into the acquisition manifest.

    Args:
        collection_ligands (DataFrame): columns collection_key, ligand_id. SEAM: the caller obtains
            this from the collection tarball file lists at COLLECTION_PREFIX (tar tzf, no extraction)
            or a stored ingestion manifest. Confirm the canonical source with the AFVS owner.
        smiles_store_path (str): the ligands store parquet(s) with ligand_id + smiles (validated:
            s3://tamarind-ligands/parquet/<dataset>/).
    """
    store = pd.read_parquet(smiles_store_path, columns=["ligand_id", "smiles"])
    store["key"] = store.ligand_id.astype(str).str.replace(r"\.\d+$", "", regex=True)
    cl = collection_ligands.copy()
    cl["key"] = cl.ligand_id.astype(str).str.replace(r"\.\d+$", "", regex=True)
    manifest = cl.merge(store[["key", "smiles"]], on="key", how="inner")[
        ["collection_key", "ligand_id", "smiles"]]
    manifest.to_parquet(out_path)
    return manifest


# ---------- SEAM B: docked scores ----------

def load_docked_scores(scores_glob):
    """
    Read accumulated per-workunit parquets. Returns (valid_scores, attempted_ligands):
      valid_scores    -> DataFrame(ligand, collection_key, score_min) for USABLE dockings (train on these).
      attempted_ligands -> set of every ligand_id with a docking record, valid OR failed. Used as the
                          selection exclusion set so we never re-dock a known-failed ligand (which would
                          re-rank the same and waste budget every round).
    """
    files = glob.glob(scores_glob, recursive=True)
    if not files:
        return pd.DataFrame(columns=["ligand", "collection_key", "score_min"]), set()
    df = pd.concat([pd.read_parquet(f, columns=["ligand", "collection_key", "score_min"])
                    for f in files], ignore_index=True)
    attempted = set(df.ligand.astype(str))
    valid = df[(df.score_min < 0) & (df.score_min > -25)]  # drop failed/clash dockings
    return valid, attempted


# ---------- round logic (testable) ----------

def run_round(manifest, docked_scores, fp_cache, budget_total, per_round, k_frac=0.001,
              patience=3, conv_eps=0.01, epsilon_random=0.0, min_spearman=0.30, seed=42,
              history=None, model_out=None, todo_out=None, granularity="molecule",
              attempted_ligands=None, max_explore_rounds=3):
    """
    One AL round on already-loaded data. Trains the surrogate on the docked scores, checks
    budget/convergence, applies the explore-then-exploit quality gate (guardrail 2), and selects
    the next batch.

    Returns (status, next_selected, info) where status in {CONTINUE, CONVERGED, BUDGET, QUALITY_LOW,
    EXHAUSTED}; info["mode"] is "explore" (random selection, surrogate not yet trusted) or "exploit"
    (greedy). Any status other than CONTINUE tells the run driver to stop the loop.
    """
    # NB: `is None`, not `or []` -- an empty [] passed by _cmd_round is the SAME object it re-dumps to
    # history.json, so appends here must mutate it in place. `history or []` would fork a new list and the
    # persisted history would stay [] forever, silently killing the explore-cap + convergence early-stops.
    if history is None:
        history = []
    smi_to_row, X = fp_cache
    total_ligands = manifest.ligand_id.nunique()

    # join docked ligand scores to fingerprints (via manifest smiles)
    lig2smi = dict(zip(manifest.ligand_id.astype(str), manifest.smiles.astype(str)))
    d = docked_scores.copy()
    d["smiles"] = d.ligand.astype(str).map(lig2smi)
    d = d.dropna(subset=["smiles"]).drop_duplicates("ligand")
    rows = d.smiles.map(smi_to_row)
    d = d[rows.notna()]
    Xtr = X[rows.dropna().astype(int).to_numpy()]
    ytr = d.score_min.to_numpy(dtype=np.float32)

    docked_collections = sorted(set(docked_scores.collection_key.astype(str)))
    docked_ligands = len(set(docked_scores.ligand.astype(str)))

    if len(ytr) < 10:
        raise ValueError(f"too few docked scores to train ({len(ytr)})")

    # final model for selection: trained on ALL docked data.
    model, meta = train_on_X(Xtr, ytr, seed=seed)

    # guardrail-2 quality = a PROPER held-out Spearman: train a probe on a train split, evaluate on the
    # held-out split. (The earlier estimate evaluated the full-data model on a subset of its OWN training
    # points -- in-sample -- which overstates quality so QUALITY_LOW / explore rarely fired.)
    # rho stays NaN only when the probe could NOT be measured (no scipy / holdout too small) -> "can't
    # measure" -> proceed greedy. If the probe RAN but scipy returned NaN, that is NOT "unmeasurable": it
    # means the probe's predictions (or the held-out scores) had zero variance, i.e. a degenerate/collapsed
    # surrogate whose ranking is worthless. Treat that as the WORST quality (-1.0), not as trusted, so it
    # trips explore + the QUALITY_LOW fallback instead of silently exploiting a useless model.
    rho = float("nan")
    if spearmanr is not None and len(ytr) >= 20:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(ytr))
        n_hold = max(3, len(ytr) // 5)                       # ~20% held out
        hold, tr = perm[:n_hold], perm[n_hold:]
        if len(tr) >= 10:
            probe, pmeta = train_on_X(Xtr[tr], ytr[tr], seed=seed)
            rho = spearmanr(predict_on_X(probe, pmeta, Xtr[hold]), ytr[hold]).correlation
            if rho is None or np.isnan(rho):
                rho = -1.0                                   # probe ran but degenerate -> worst quality
    if model_out:
        save_regressor(model, meta, model_out)

    # convergence on the rolling mean of the top-K docked score
    k = max(1, int(round(total_ligands * k_frac)))
    topk_mean = float(np.sort(ytr)[:k].mean())
    history.append({"round": len(history), "docked_ligands": docked_ligands,
                    "topk_mean": topk_mean, "spearman": rho})
    converged = False
    if len(history) > patience:
        prev = np.mean([h["topk_mean"] for h in history[-patience - 1:-1]])
        if prev != 0 and abs(topk_mean - prev) / abs(prev) < conv_eps:
            converged = True

    # guardrail 2 (explore-then-exploit): trust the surrogate's ranking only when the held-out Spearman
    # clears min_spearman. A bad seed surrogate usually just needs more/broader training data, so instead
    # of giving up we dock a RANDOM (unbiased) batch, retrain, and re-check next round -- NOT full brute
    # force, always bounded by budget_total. Once Spearman recovers we switch to greedy exploitation. If
    # it stays low for max_explore_rounds consecutive rounds the target is effectively unlearnable from
    # fingerprints, so we stop (QUALITY_LOW). NaN Spearman here means ONLY "unmeasurable" (no scipy /
    # holdout too small) -> proceed greedy; a degenerate probe was already mapped to -1.0 above.
    trusted = np.isnan(rho) or rho >= min_spearman
    bad_streak = 0                              # consecutive low-quality rounds, most-recent first
    for h in reversed(history):
        hs = h.get("spearman")
        if hs is None or (isinstance(hs, float) and np.isnan(hs)) or hs >= min_spearman:
            break
        bad_streak += 1

    # budget is a compute cap: count every ATTEMPTED docking (valid + failed), not just the valid ones,
    # so dock failures don't let the loop overspend the customer's docking budget.
    n_attempted = len(attempted_ligands) if attempted_ligands is not None else docked_ligands
    info = {"docked_ligands": docked_ligands, "attempted": n_attempted, "topk_mean": topk_mean,
            "spearman": rho, "n_docked_collections": len(docked_collections),
            "mode": "exploit" if trusted else "explore"}

    if n_attempted >= budget_total:
        return "BUDGET", [], info
    if not trusted:
        if bad_streak >= max_explore_rounds:
            return "QUALITY_LOW", [], info       # surrogate never became reliable -> stop early
        explore = True                           # surrogate not yet trustworthy: sample randomly
    else:
        if converged:
            return "CONVERGED", [], info
        explore = False

    # acquire the next batch (MolPAL `random` when exploring, MolPAL `greedy` when exploiting)
    remaining = budget_total - n_attempted
    this_round = min(per_round, remaining)
    # exclude ALL attempted ligands (valid + failed) so a known-failed dock isn't re-selected.
    exclude = attempted_ligands if attempted_ligands is not None \
        else set(docked_scores.ligand.astype(str))
    if granularity == "molecule":
        # PRIMARY path: rank individual ligands, emit an AFVS csv_collection_key_ligand list
        # (named mode). Recall 0.75 vs 0.14 for collection-granular on real qvina02 data.
        # Acquisition is MolPAL's, always. alAcquirer only picks WHICH MolPAL metric; unset means
        # MolPAL's own default, greedy. The explore round is MolPAL's `random` metric rather than
        # its `epsilon`, which draws over the whole pool including already-explored ligands.
        # Validate BEFORE spending anything. An unsupported metric otherwise raises inside MolPAL's
        # metrics.calc on the first EXPLOIT round, i.e. after the seed and every explore round of the
        # customer's docking budget is already spent, and `threshold` does not raise at all: it
        # returns a constant, which is a fully random screen at full cost.
        if acq_metric not in SUPPORTED_METRICS:
            raise ValueError(
                "alAcquirer=%r is not supported. Use one of: %s"
                % (acq_metric, ", ".join(sorted(SUPPORTED_METRICS))))
        # MolPAL's uncertainty metrics need a per-ligand variance. Our FNN is a point predictor, so
        # without an ensemble they all silently collapse to greedy. Train one only when a metric
        # actually consumes the variance: it costs n_models x training for no benefit otherwise.
        n_ens = max(1, int(env_setting("alEnsemble") or 1))
        if acq_metric in VAR_METRICS and n_ens > 1 and not explore:
            handles = train_ensemble_on_X(Xtr, ytr, n_models=n_ens, seed=seed)
            mp = predict_ligand_scores_ensemble(handles, manifest, fp_cache=fp_cache)
            pred_var = mp["pred_var"].to_numpy()
        else:
            mp = predict_ligand_scores(model, meta, manifest, fp_cache=fp_cache)
            pred_var = None
        # Thread the REAL docked scores through: ei and pi read them as the incumbent to improve on.
        # Passing only the exclusion set leaves that incumbent at -inf and silently inverts them.
        selected = select_ligands(mp, exclude, this_round, metric=acq_metric,
                                  explore=explore, seed=seed, pred_var=pred_var,
                                  docked_scores=dict(zip(docked_scores.ligand.astype(str),
                                                         docked_scores.score_min.astype(float))))
        if not len(selected):
            return "EXHAUSTED", [], info          # pool fully attempted -> stop (don't re-dock a stale wave)
        if todo_out:
            write_named_csv(selected, todo_out)
        return "CONTINUE", list(selected.ligand_id.astype(str)), info
    # collection-granular option (only viable for a genuinely property-tranched giga-library)
    agg = per_collection_scores(model, meta, manifest, fp_cache=fp_cache)
    # collection granularity keeps its own epsilon-random selector: it accumulates until a LIGAND
    # budget is met rather than taking a fixed top-k, so MolPAL's Acquirer is not a fit for it.
    selected = select_collections(agg, docked_collections, this_round,
                                  1.0 if explore else epsilon_random, seed)
    if not selected:
        return "EXHAUSTED", [], info
    if todo_out:
        write_todo(agg, selected, todo_out)
    return "CONTINUE", selected, info


# ---------- CLI ----------

def _read_one_tarball_members(args):
    """Worker: return [(collection_key, ligand_id), ...] for one collection tarball (file list only,
    no extraction). Module-level so it's picklable for a process pool."""
    import subprocess
    import tarfile
    import io
    coll_key, data_bucket, s3key = args
    try:
        raw = subprocess.run(["aws", "s3", "cp", f"s3://{data_bucket}/{s3key}", "-"],
                             capture_output=True).stdout
        out = []
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
            for name in tf.getnames():
                if "/" not in name or name.endswith("/"):   # skip the collection dir entry itself
                    continue
                base = os.path.basename(name)               # 177154706.pdbqt
                out.append((coll_key, os.path.splitext(base)[0]))
        return out
    except Exception as e:
        print(f"WARN: failed reading tarball {s3key}: {e}", file=sys.stderr)
        return []


def collection_ligands_from_tarballs(collection_prefix, todo_all_path, data_bucket, cache_path=None,
                                     max_workers=32):
    """
    SEAM A. Build the (collection_key -> ligand_id) table by reading collection tarball FILE LISTS
    (no extraction). Collections live at s3://<data_bucket>/<collection_prefix>/... under AFVS's
    metatranche addressing (<prefix>/<metatranche>/<tranche>/<collection>.tar.gz).

    Scale note (the giga-scale bottleneck this fixes): a large library has O(10k+) collections
    (ChEMBL-36 = 12,699), and reading them SERIALLY was ~3-7h before any docking. Two fixes here:
      1. CACHE: this map is DATASET-STATIC, so persist it to `cache_path` (a parquet next to the AL
         state) and reuse it on any later run of the same dataset -- the scan happens at most once.
      2. PARALLELIZE: the tarball reads are independent network I/O, so fan them out across a thread
         pool (max_workers) instead of one-at-a-time. ~30x fewer wall-clock seconds on a 12.7k library.
    A stored ingestion-time ligand->collection index would remove the scan entirely (pass
    --collection-ligands); until ingestion emits one, the cached parallel scan is the fix.
    """
    import concurrent.futures
    import subprocess

    if cache_path and os.path.exists(cache_path):
        print(f"SEAM-A: reusing cached collection->ligand map at {cache_path}", file=sys.stderr)
        return pd.read_parquet(cache_path)

    # one recursive listing -> collection_key -> full S3 key
    listing = subprocess.run(
        ["aws", "s3", "ls", f"s3://{data_bucket}/{collection_prefix}/", "--recursive"],
        capture_output=True, text=True).stdout
    key_by_coll = {}
    for row in listing.splitlines():
        cols = row.split()
        if not cols or not cols[-1].endswith(".tar.gz"):
            continue
        parts = cols[-1].split("/")
        coll = parts[-1][:-len(".tar.gz")]                 # 00000000 (.tar.gz is a double ext)
        tranche = parts[-2] if len(parts) >= 2 else ""     # AA01
        key_by_coll[f"{tranche}_{coll}"] = cols[-1]

    # the collections we actually need = those in the todo.all
    work = []
    for line in open(todo_all_path):
        coll_key = line.strip().split()[0] if line.strip() else ""
        if not coll_key:
            continue
        s3key = key_by_coll.get(coll_key)
        if not s3key:
            print(f"WARN: collection {coll_key} not found under {collection_prefix}", file=sys.stderr)
            continue
        work.append((coll_key, data_bucket, s3key))

    print(f"SEAM-A: reading {len(work)} collection tarballs (parallel, {max_workers} workers)...",
          file=sys.stderr)
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for res in ex.map(_read_one_tarball_members, work):
            rows.extend(res)
    df = pd.DataFrame(rows, columns=["collection_key", "ligand_id"])
    if cache_path:
        try:
            df.to_parquet(cache_path)
            print(f"SEAM-A: cached collection->ligand map to {cache_path} ({len(df)} rows)", file=sys.stderr)
        except Exception as e:
            print(f"WARN: failed to cache collection map: {e}", file=sys.stderr)
    return df


def _cmd_init(args):
    from ml_regressor import fingerprint_all
    st = _state(args.state_dir)
    os.makedirs(args.state_dir, exist_ok=True)

    # SEAM A: (collection_key, ligand_id). Prefer a pre-built ingestion manifest; else scan the tarball
    # file lists (parallel + cached to the state dir so the O(10k)-collection scan happens at most once).
    if args.collection_ligands:
        cl = pd.read_parquet(args.collection_ligands)
    else:
        cl = collection_ligands_from_tarballs(args.collection_prefix, args.todo, args.data_bucket,
                                              cache_path=os.path.join(args.state_dir, "collection_ligands.parquet"))
    manifest = build_manifest(cl, args.smiles_store, st["manifest"])

    # Fingerprint the pool once -> cache reused across all rounds. Use fingerprint_all so a bad/
    # unparseable SMILES is DROPPED from the cache (not stored as an all-zero row): the cache's
    # smi_to_row is built from these `smiles`, so an unlisted SMILES resolves to row -1 at select
    # time and sinks to +inf (unscoreable), matching the fingerprint_all fallback path. Storing a
    # zero-vector row instead would silently score a bad ligand and could waste budget / skew a
    # collection's predicted-best. Dropping bad rows also shrinks the cached matrix.
    all_smiles = manifest.smiles.astype(str).tolist()
    kept, X = fingerprint_all(all_smiles)
    smiles = [all_smiles[i] for i in kept]
    np.savez(st["fp"], smiles=np.array(smiles, dtype=object), X=X)

    # seed: a random subset of LIGANDS, written as an AFVS csv_collection_key_ligand list (named mode).
    # Per-molecule is the primary path (recall 0.78 vs 0.14 collection-granular; collection-granularity-finding.md).
    # seed = seed_frac of the pool, floored at min_seed so a small library still docks enough to train a
    # surrogate (round 1 needs >= 10 valid dockings; the floor absorbs the docking failure rate), CAPPED at
    # the pool size AND at the docking budget: the seed docks unconditionally in wave 1 and the budget gate
    # only fires afterward, so a budget-blind seed would overspend (e.g. seed_frac*pool > alBudget) -- clamp
    # it here so the seed never exceeds what the customer authorized.
    pool = manifest.ligand_id.nunique()
    seed_target = max(int(round(pool * args.seed_frac)), args.min_seed)
    seed_target = min(seed_target, pool)
    if args.budget > 0:
        seed_target = min(seed_target, args.budget)
    seed_ligs = manifest.sample(seed_target, random_state=args.seed)
    n_seed = write_named_csv(seed_ligs, args.out)
    json.dump([], open(st["history"], "w"))
    print(f"INIT manifest={len(manifest)} ligands / {manifest.collection_key.nunique()} collections; "
          f"seed={n_seed} ligands (budget={args.budget}) -> {args.out}")
    return "INIT"


def _cmd_round(args):
    st = _state(args.state_dir)
    manifest = pd.read_parquet(st["manifest"])
    fp_cache = _load_fp_cache(st["fp"])
    docked, attempted = load_docked_scores(args.scores_glob)
    history = json.load(open(st["history"])) if os.path.exists(st["history"]) else []
    status, selected, info = run_round(
        manifest, docked, fp_cache, args.budget, args.per_round, k_frac=args.k_frac,
        patience=args.patience, epsilon_random=args.epsilon_random, min_spearman=args.min_spearman,
        seed=args.seed, history=history, model_out=st["model"], todo_out=args.out,
        attempted_ligands=attempted, max_explore_rounds=args.max_explore_rounds)
    # Enrich the round entry run_round just appended with mode + terminal status so history.json is the
    # single source of truth for the website progress panel (the run driver publishes it verbatim to
    # the DDB ALProgress map). mode/status aren't known inside run_round at append time.
    if history:
        history[-1]["mode"] = info.get("mode", "-")
        history[-1]["status"] = status
    json.dump(history, open(st["history"], "w"))
    print(f"STATUS={status} mode={info.get('mode', '-')} docked_ligands={info['docked_ligands']} "
          f"topk_mean={info['topk_mean']:.3f} spearman={info['spearman']:.3f} "
          f"selected_ligands={len(selected)}")
    return status


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("round", "init"):
        ap = argparse.ArgumentParser(description=__doc__)
        sub = ap.add_subparsers(dest="cmd")
        r = sub.add_parser("round")
        r.add_argument("--state-dir", required=True)
        r.add_argument("--scores-glob", required=True)
        r.add_argument("--out", required=True, help="next todo.all path")
        r.add_argument("--budget", type=int, required=True)
        r.add_argument("--per-round", type=int, required=True)
        r.add_argument("--k-frac", type=float, default=0.001)
        r.add_argument("--patience", type=int, default=3)
        r.add_argument("--epsilon-random", type=float, default=0.0)  # MolPAL default; collection path only
        r.add_argument("--min-spearman", type=float, default=0.30)
        r.add_argument("--max-explore-rounds", type=int, default=3,
                       help="consecutive low-Spearman rounds to random-explore before stopping (QUALITY_LOW)")
        r.add_argument("--seed", type=int, default=42)
        i = sub.add_parser("init")
        i.add_argument("--state-dir", required=True)
        i.add_argument("--smiles-store", required=True)
        i.add_argument("--out", required=True, help="seed todo.all path")
        i.add_argument("--collection-ligands", default="", help="pre-built parquet: collection_key, ligand_id")
        i.add_argument("--collection-prefix", default="")
        i.add_argument("--data-bucket", default="tamarind-ligands")
        i.add_argument("--todo", default="", help="full-library todo.all (for the tarball manifest seam)")
        i.add_argument("--seed-frac", type=float, default=0.01)
        i.add_argument("--min-seed", type=int, default=30, help="floor on the seed size for small libraries")
        i.add_argument("--budget", type=int, default=0, help="docking budget; caps the seed so it can't overspend (0 = no cap)")
        i.add_argument("--seed", type=int, default=42)
        args = ap.parse_args()
        if args.cmd == "round":
            _cmd_round(args)
        elif args.cmd == "init":
            _cmd_init(args)
        sys.exit(0)

    # --- self-test: the round logic on synthetic manifest + synthetic docked scores ---
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    demo = ['CCO', 'CCCO', 'CCCCO', 'CCN', 'CCCN', 'c1ccccc1', 'c1ccccc1C', 'c1ccccc1O', 'CC(=O)O',
            'CC(=O)N', 'c1ccncc1', 'C1CCCCC1', 'CC(C)C', 'CCCCCCCCCC', 'c1ccc2ccccc2c1',
            'NC(Cc1ccccc1)C(=O)O', 'OC1CCCCC1', 'FC1=CC=CC=C1', 'CCOCC', 'CCC(=O)CC',
            'c1ccc(cc1)C(=O)O', 'CC(=O)Nc1ccccc1', 'COc1ccccc1', 'NCCc1ccccc1', 'CC(C)Oc1ccccc1',
            'CCCCCCCCCCCC', 'c1ccc2cc3ccccc3cc2c1', 'CC1CCCCC1', 'OCCc1ccccc1', 'ClC1=CC=CC=C1'] * 2
    rng = np.random.RandomState(0)
    truescore = {s: -1.0 * Descriptors.HeavyAtomCount(Chem.MolFromSmiles(s)) for s in set(demo)}

    rows, cols = [], {}
    for i, s in enumerate(demo):
        ck = f"AA_{i // 5:07d}"
        rows.append({"collection_key": ck, "ligand_id": f"L{i}", "smiles": s})
    manifest = pd.DataFrame(rows).drop_duplicates("ligand_id")

    from ml_regressor import smi_to_fingerprint
    Xfull = np.zeros((len(manifest), 1024), dtype=np.float32)
    smi_to_row = {}
    for i, s in enumerate(manifest.smiles.tolist()):
        fp = smi_to_fingerprint(s)
        if fp is not None:
            Xfull[i] = fp
        smi_to_row[s] = i
    fp_cache = (smi_to_row, Xfull)

    # "dock" the first 8 collections (40 ligands) -> synthetic scores (enough for a >2 holdout)
    docked_ck = [f"AA_{c:07d}" for c in range(8)]
    dm = manifest[manifest.collection_key.isin(docked_ck)]
    docked = pd.DataFrame({"ligand": dm.ligand_id.values, "collection_key": dm.collection_key.values,
                           "score_min": [truescore[s] + rng.normal(0, 1.0) for s in dm.smiles]})

    status, selected, info = run_round(manifest, docked, fp_cache, budget_total=60, per_round=10,
                                       k_frac=0.1, history=[], seed=0)
    docked_lig_ids = set(docked.ligand)
    assert status == "CONTINUE", f"expected CONTINUE, got {status}"
    assert info["mode"] == "exploit", f"good surrogate should exploit, got {info['mode']}"
    assert selected and all(l not in docked_lig_ids for l in selected), "must select fresh ligands"
    assert info["spearman"] == info["spearman"], "spearman should be computed on a >2 holdout"
    print(f"self-test PASSED (per-molecule): status={status}, mode={info['mode']}, selected {len(selected)} "
          f"fresh ligands, spearman={info['spearman']:.2f}, topk_mean={info['topk_mean']:.2f}")

    # explore-then-exploit: force "untrusted" with an unreachable min_spearman -> should EXPLORE (random
    # select + CONTINUE), not stop, while the bad-streak is under the cap.
    st2, sel2, info2 = run_round(manifest, docked, fp_cache, budget_total=60, per_round=10,
                                 k_frac=0.1, history=[], seed=0, min_spearman=0.99)
    assert st2 == "CONTINUE" and info2["mode"] == "explore", f"bad surrogate -> explore, got {st2}/{info2['mode']}"
    assert sel2 and all(l not in docked_lig_ids for l in sel2), "explore round must still select fresh ligands"
    # cap: two prior low-quality rounds + this one -> bad_streak 3 -> QUALITY_LOW stop
    badhist = [{"round": 0, "spearman": 0.0, "docked_ligands": 10, "topk_mean": -1.0},
               {"round": 1, "spearman": 0.0, "docked_ligands": 20, "topk_mean": -1.0}]
    st3, _, _ = run_round(manifest, docked, fp_cache, budget_total=60, per_round=10, k_frac=0.1,
                          history=badhist, seed=0, min_spearman=0.99, max_explore_rounds=3)
    assert st3 == "QUALITY_LOW", f"explore cap should stop with QUALITY_LOW, got {st3}"
    print("self-test PASSED (explore-then-exploit): explore selects randomly + CONTINUE; cap stops QUALITY_LOW")

    # history persistence: run_round MUST append to the caller's list in place (not fork it), else the
    # cross-round explore-cap + convergence early-stops are dead in the real _cmd_round dispatch path.
    hist = []
    run_round(manifest, docked, fp_cache, budget_total=60, per_round=10, k_frac=0.1, history=hist, seed=0)
    assert len(hist) == 1, f"run_round must append to the passed-in history in place, got len={len(hist)}"
    run_round(manifest, docked, fp_cache, budget_total=60, per_round=10, k_frac=0.1, history=hist, seed=0)
    assert len(hist) == 2, f"history must accumulate across rounds, got len={len(hist)}"
    print("self-test PASSED (history persistence): run_round mutates the caller's history in place")
