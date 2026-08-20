#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression surrogate for multi-round active-learning virtual screening (TAM-1599).

The regression counterpart of ml_classifier.py. The network BODY (1024 -> 512 -> 256 -> 128, ReLU)
and Morgan-1024 featurization are kept verbatim from the AdaptiveFlow manuscript classifier; the
Sigmoid classification head + BCE loss + top-percent labeling are REPLACED by a linear regression
head + Huber loss on the raw docking score. This gives "regress-and-rank" acquisition (predict the
continuous docking score, rank by it) rather than the lossy classify-and-prune: a collection passed
over early can be selected in a later round if the improved model ranks it up.

Score convention: more negative = better binder (kcal/mol). Targets are standardized for stable
training and inverted on predict, so predictions come back in the original score units.

Validated against exhaustively-docked ground truth: 5-round recall of the true top-1% is
~0.75-0.96 at 10% docked across Enamine (Vina), ChEMBL (our qvina02), and 58 dockstring targets;
the held-out Spearman gate reliably flags low-recall targets (guardrail 2).

Shared by the AL controller (afvs_al_controller.py, which trains it on accumulated docking scores)
and the acquisition step (afvs_al_select.py, which ranks collections by predicted score). Operates on
whatever (SMILES, score) pairs it is given; it has no knowledge of tranches, collections, or joblines.
"""
import os
import random
import uuid

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def get_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Featurization is MolPAL's atom-pair, adopted by MEASUREMENT rather than by preference.
#
# Gate 1 (2026-08-13, 58 DOCKSTRING targets x 3 seeds, 1,102 campaigns, shuffled-label control
# passed) compared four featurizers under one FNN, paired per target against the Morgan baseline:
#   MolPAL pair maxLength 3 @ 2048  +0.0357 top-1% recall, p=3.2e-08, 51/58 targets  <- adopted
#   our atom-pair maxDistance 30 @ 1024  +0.0333, p=2.9e-08, 48/58
#   MolPAL pair maxLength 3 @ 1024  +0.0283, p=2.8e-07, 49/58
# against a seed spread of 0.0270. The first two are statistically indistinguishable (0.0024 apart),
# so this is MolPAL's parameterization on provenance grounds as much as on the number.
#
# The SURROGATE is deliberately NOT MolPAL's: the same run measured MolPAL's RF at -0.1140 and its
# GP at -0.1037 against this FNN, each winning 0 of 58 targets. Do not "finish the job" by swapping
# the model too; that was measured and rejected.
#
# Provenance, stated honestly: this is an rdkit atom-pair fingerprint parameterized to be
# BIT-IDENTICAL to MolPAL's, asserted 200/200 molecules at both widths as a fail-closed gate on every
# Gate 1 run. It is NOT MolPAL's code executing. An earlier version imported MolPAL's own Featurizer
# with this as a fallback and stamped which path ran into the model metadata, but that import can
# never resolve in any deployed configuration: molpal is not installed on the head node and cannot be
# (its dependency set conflicts with the box's numpy), and it is not vendorable either because
# molpal/featurizer.py imports ray, which is the exact dependency vendoring the Acquirer exists to
# keep off this machine. So the stamp could only ever emit one value and certified nothing. The
# equivalence argument is the real claim; do not dress it up as a provenance fact.
DEFAULT_FP_TYPE = "molpal_pair"
DEFAULT_N_BITS = 2048


def smi_to_fingerprint(smi, radius=2, n_bits=None, fp_type=None):
    """SMILES -> fingerprint, or None if unparseable/empty.

    fp_type "molpal_pair" (default) is MolPAL's atom-pair; "morgan" is the pre-2026-08-13 default,
    kept so an old checkpoint can be re-featurized rather than silently mis-scored.
    """
    n_bits = DEFAULT_N_BITS if n_bits is None else n_bits
    fp_type = DEFAULT_FP_TYPE if fp_type is None else fp_type
    if not smi or smi in ("N/A", "None", "nan"):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    if fp_type == "morgan":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        for bit in fp.GetOnBits():
            arr[bit] = 1.0
        return arr

    if fp_type != "molpal_pair":
        raise ValueError("unknown fp_type %r (expected 'molpal_pair' or 'morgan')" % (fp_type,))

    # minLength/maxLength here ARE the equivalence: MolPAL builds its "pair" fingerprint as
    # GetHashedAtomPairFingerprintAsBitVect(minLength=1, maxLength=1+radius), so these two bounds
    # are what make the bit-identity claim true. Changing either silently breaks it.
    fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
        mol, minLength=1, maxLength=1 + radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    ConvertToNumpyArray(fp, arr)
    return arr


def fingerprint_all(smiles_list, radius=2, n_bits=None, fp_type=None):
    """
    Fingerprint a list of SMILES, dropping any that fail to parse.
    Returns (kept_indices, X) where kept_indices maps X rows back to smiles_list positions.

    Materializes the whole float32 matrix in RAM (n_bits*4 B/molecule, plus an equal-sized
    intermediate list before the stack, so ~2x at peak). That is fine for a docked TRAINING set,
    which is budget-bounded, and is why this stays the wrapper used by train_regressor /
    predict_scores. For a full SCREENING POOL use fingerprint_to_packed_npy below, which is 32x
    smaller and streams; at 2048 bits this function peaks near 16.7 kB/molecule and is what caps
    the pool just under 1e6 on a 16 GiB head node.
    """
    n_bits = DEFAULT_N_BITS if n_bits is None else n_bits
    fp_type = DEFAULT_FP_TYPE if fp_type is None else fp_type
    kept_indices, fps = [], []
    for i, smi in enumerate(smiles_list):
        fp = smi_to_fingerprint(smi, radius=radius, n_bits=n_bits, fp_type=fp_type)
        if fp is not None:
            kept_indices.append(i)
            fps.append(fp)
    if not fps:
        return [], np.zeros((0, n_bits), dtype=np.float32)
    return kept_indices, np.stack(fps, axis=0)


# ---- packed fingerprints (the screening-pool representation) ----
#
# The fingerprint is BINARY but smi_to_fingerprint returns it as float32, a 32x waste that is
# invisible at the 3e4 ligands validated end to end and fatal at 1e8. np.packbits collapses it to
# n_bits/8 bytes per molecule (256 B at 2048 bits) and round-trips EXACTLY, so this is a
# representation change and not an approximation. Bit order is numpy's default ('big'); pack and
# unpack must agree, which is why neither call site passes bitorder.
#
# Why a .npy and not the .npz the cache used to be: np.load(..., mmap_mode='r') SILENTLY IGNORES
# mmap_mode for an .npz and returns a fully-resident array with OWNDATA True, no error and no
# warning. So the obvious "just memory-map the cache" fix is a no-op on the old format. A plain
# .npy memory-maps for real (OWNDATA False), which is what lets the pool exceed RAM.

def packed_width(n_bits):
    """Bytes per molecule in the packed representation."""
    if n_bits % 8:
        raise ValueError("n_bits must be a multiple of 8 to pack, got %r" % (n_bits,))
    return n_bits // 8


def pack_fp(X):
    """float32/uint8 0-1 matrix (n, n_bits) -> packed uint8 (n, n_bits/8)."""
    return np.packbits(np.asarray(X).astype(np.uint8), axis=1)


def unpack_fp(P, n_bits=None):
    """Packed uint8 (n, n_bits/8) -> float32 (n, n_bits). The inverse of pack_fp."""
    out = np.unpackbits(np.asarray(P, dtype=np.uint8), axis=1).astype(np.float32)
    return out if n_bits is None else out[:, :n_bits]


def is_packed(X, n_bits):
    """True when X is the packed representation of an n_bits fingerprint.

    Dispatching on dtype is safe here because BOTH mis-dispatches fail loudly rather than
    silently: feeding a packed matrix to a model expecting n_bits raises a shape error in the
    first nn.Linear, and np.unpackbits on a float32 array raises a TypeError. Neither can produce
    a plausible-but-wrong prediction, which is the failure mode that matters for an acquisition
    ranking.
    """
    return getattr(X, "dtype", None) == np.uint8 and X.shape[1] == packed_width(n_bits)


def fingerprint_to_packed_npy(smiles_list, out_path, radius=2, n_bits=None, fp_type=None,
                              chunk=50_000, progress_every=0):
    """
    Stream SMILES -> a packed uint8 .npy on disk, never holding the full matrix in RAM.

    Returns (kept_indices, X) where X is a memory-mapped VIEW of just the kept rows, so callers
    index it exactly like the fingerprint_all result. Peak RAM is one chunk (chunk * n_bits/8
    bytes, ~12.8 MB at the default) plus kept_indices, instead of ~2x the whole float32 matrix.

    Unparseable SMILES are DROPPED, matching fingerprint_all: the cache's row order is the kept
    order, and an unlisted SMILES resolves to row -1 at select time and sinks to +inf. The file is
    allocated for len(smiles_list) rows and returned sliced to the kept count, so a parse-failure
    rate f leaves f of the file unused rather than paying a full second copy to trim it; at the
    sub-1% rates observed that is cheaper than the rewrite.
    """
    n_bits = DEFAULT_N_BITS if n_bits is None else n_bits
    fp_type = DEFAULT_FP_TYPE if fp_type is None else fp_type
    width = packed_width(n_bits)
    n = len(smiles_list)

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8,
                                   shape=(max(n, 1), width))
    kept_indices = []
    buf = np.zeros((chunk, width), dtype=np.uint8)
    b = 0   # rows buffered
    k = 0   # rows written

    def _flush():
        nonlocal b, k
        if b:
            mm[k:k + b] = buf[:b]
            k += b
            b = 0

    for i, smi in enumerate(smiles_list):
        fp = smi_to_fingerprint(smi, radius=radius, n_bits=n_bits, fp_type=fp_type)
        if fp is None:
            continue
        buf[b] = np.packbits(fp.astype(np.uint8))
        b += 1
        kept_indices.append(i)
        if b == chunk:
            _flush()
            if progress_every and k % progress_every == 0:
                print(f"  fingerprinted {k:,}/{n:,}", flush=True)
    _flush()

    mm.flush()
    return kept_indices, mm[:k]


def load_packed_npy(path, mmap=True):
    """Load a packed fingerprint cache. mmap=True keeps it on disk (OWNDATA False)."""
    return np.load(path, mmap_mode="r" if mmap else None)


class FNNRegressor(nn.Module):
    """Body matches the manuscript classifier (1024->512->256->128, ReLU); linear head (no sigmoid)."""

    def __init__(self, input_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_on_X(X, y, val_fraction=0.10, batch_size=256, max_epochs=100, patience=8,
               learning_rate=1e-3, huber_delta=1.0, seed=42, device=None, verbose=False):
    """
    Train the FNN regressor on a precomputed fingerprint matrix X and aligned scores y.
    Returns (model, meta); meta carries the score standardization (mean/std) + training history.
    """
    set_seed(seed)
    device = get_device(device)
    # A packed matrix is unpacked WHOLE here, unlike predict_on_X which unpacks per chunk. That is
    # deliberate and bounded: this only ever sees the DOCKED set, which is capped by alBudget, so
    # it is 7% of pool at the default budget shape rather than the pool itself. It is still the
    # next ceiling to fall on this path (at 1e8 pool that is 7e6 rows ~ 57 GB unpacked), so a
    # per-batch unpacking Dataset is the fix when the pool goes past ~1e7, not before.
    if getattr(X, "dtype", None) == np.uint8:
        X = unpack_fp(X, X.shape[1] * 8)
    else:
        X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    y_mean, y_std = float(y.mean()), float(y.std())
    if y_std < 1e-8:
        y_std = 1.0
    y_z = (y - y_mean) / y_std

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y_z))
    n_val = max(1, int(round(len(y_z) * val_fraction)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = TensorDataset(torch.from_numpy(X[train_idx]),
                             torch.from_numpy(y_z[train_idx].astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    Xv = torch.from_numpy(X[val_idx]).to(device)
    yv = torch.from_numpy(y_z[val_idx].astype(np.float32)).to(device)

    model = FNNRegressor(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.HuberLoss(delta=huber_delta)

    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, best_epoch, no_improve = float("inf"), None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb).squeeze(1), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xv).squeeze(1), yv).item())
        history["train_loss"].append(float(np.mean(losses)))
        history["val_loss"].append(val_loss)
        if verbose:
            print(f"  epoch {epoch + 1}/{max_epochs}  val {val_loss:.4f}")
        if val_loss < best_val:
            best_val, best_epoch, no_improve = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    # fp_type is stamped from the module default because train_on_X only ever sees a MATRIX, not
    # the SMILES that produced it. A caller that featurized with a non-default fp_type must
    # override it (train_regressor does). Recorded so predict_scores cannot silently re-featurize
    # a checkpoint with a different fingerprint, which inverts the ranking with no error.
    meta = {"input_dim": X.shape[1], "y_mean": y_mean, "y_std": y_std,
            "best_epoch": best_epoch, "best_val_loss": best_val,
            "fp_radius": 2, "fp_nbits": X.shape[1],
            "fp_type": DEFAULT_FP_TYPE}
    return model, meta


# Bounded in bytes of the unpacked block, not rows: the safe row count depends on the fingerprint
# width, and that width has already moved once (1024 -> 2048) without the row-count default moving.
DEFAULT_CHUNK_BYTES = 128 << 20  # 128 MB unpacked float32 -> 16,384 rows at 2048 bits


def _rows_per_chunk(n_bits, chunk):
    """Rows per chunk: an explicit caller-supplied `chunk` wins, else bound the block by bytes."""
    if chunk is not None:
        return int(chunk)
    return max(1, DEFAULT_CHUNK_BYTES // (int(n_bits) * 4))


def predict_on_X(model, meta, X, device=None, chunk=None, rows=None):
    """Predict docking scores (original units) for a precomputed fingerprint matrix X, chunked.

    X may be a float32 matrix OR the packed uint8 representation (including a memory-mapped one),
    in which case each chunk is unpacked as it is consumed. The materialization is deliberately
    INSIDE the loop: the previous `np.asarray(X, dtype=np.float32)` on entry pulled the entire
    pool into RAM before the first chunk, which silently defeated any memory-mapped cache handed
    to it (a memmap goes in with OWNDATA False and comes out of asarray with OWNDATA True), so the
    chunking bounded only tensor memory and not the array it was chunking.

    `rows` is an optional integer index array selecting a SUBSET of X, and it exists because the
    obvious caller-side spelling defeats everything the paragraph above buys. `X[rows]` is a
    fancy-index gather, so numpy materializes the whole selection as a new array BEFORE this
    function is entered; the chunk loop then bounds nothing, because the peak already happened in
    the caller. Passing the indices instead keeps the gather per-chunk, so peak memory is
    O(chunk) rather than O(len(rows)). At a 1e9 pool the difference is a 256 GB allocation on a
    123 GB box versus a few hundred MB. Duplicate and unsorted indices are both fine.
    """
    device = get_device(device)
    model = model.to(device)
    model.eval()
    n_bits = int(meta.get("input_dim") or meta.get("fp_nbits") or DEFAULT_N_BITS)
    chunk = _rows_per_chunk(n_bits, chunk)
    packed = is_packed(X, n_bits)
    if rows is None:
        n = X.shape[0]
    else:
        rows = np.asarray(rows)
        if rows.ndim != 1:
            raise ValueError(f"rows must be a 1-D index array, got shape {rows.shape}")
        n = int(rows.shape[0])
    out = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, chunk):
            if rows is None:
                blk = X[start:start + chunk]
                order = None
            else:
                idx = rows[start:start + chunk]
                # Sorted gather, inverted on the way out: on a memmap an arbitrary index order is
                # a random-page read. Duplicates are fine (gathered and scattered back twice).
                order = np.argsort(idx, kind="stable")
                blk = X[idx[order]]
            blk = (unpack_fp(blk, n_bits) if packed
                   else np.ascontiguousarray(blk, dtype=np.float32))
            z = model(torch.from_numpy(blk).to(device)).squeeze(1).cpu().numpy()
            z = z * meta["y_std"] + meta["y_mean"]
            # Basic slicing, so `dest` is a view into out and the scatter writes through.
            dest = out[start:start + chunk]
            if order is None:
                dest[:] = z
            else:
                dest[order] = z
    return out


# ---- uncertainty ----

# MolPAL's acquisition metrics beyond greedy (ucb / ei / pi / ts) need a per-ligand VARIANCE, and a
# point-prediction network has none: at all-zero variance every one of them returns greedy's batch.
# MolPAL solves this with NN conf_methods (dropout / mve / ensemble). We take the ENSEMBLE one, and
# deliberately not the other two, because Gate 1 measured THIS network with THIS loss as the best
# surrogate of the five tried (0 of 58 targets lost to MolPAL's RF or GP). `mve` would replace Huber
# with a Gaussian NLL and `dropout` would regularize training, and both change the model we just
# validated. An ensemble leaves the architecture and the loss untouched and gets the variance from
# disagreement between seeds.
#
# Cost is n_models x training. Training is the cheap half of a round (the expensive half is
# featurizing and predicting over the undocked pool, which is shared across the ensemble), so this
# is roughly linear in n_models on a small term.

DEFAULT_ENSEMBLE = 5


def train_ensemble_on_X(X, y, n_models=DEFAULT_ENSEMBLE, seed=42, **kwargs):
    """Train n_models independent FNNs on the same data, differing only by seed.

    Returns a list of (model, meta). Element 0 uses `seed` exactly, so a 1-model ensemble is
    bit-identical to train_on_X and the uncertainty path degrades cleanly to the measured one.
    """
    out = []
    for i in range(max(1, int(n_models))):
        out.append(train_on_X(X, y, seed=seed + i, **kwargs))
    return out


def predict_ensemble_on_X(handles, X, device=None, chunk=None, rows=None):
    """Return (mean, var) over an ensemble, both in the ORIGINAL score units.

    var is the population variance of the members' predictions, i.e. how much the ensemble disagrees
    about a ligand. It is an uncertainty SIGNAL for acquisition, not a calibrated posterior, and it
    is documented that way so nobody reads a confidence interval off it. A 1-member ensemble returns
    all-zero variance, which correctly degrades every uncertainty metric back to greedy.

    `rows` is forwarded to predict_on_X unchanged; see its docstring for why passing indices beats
    passing X[rows]. Forwarding matters more here than on the single-model path, because without it
    the caller's one materialized gather is paid once and then read by every ensemble member.
    """
    preds = np.stack([predict_on_X(m, meta, X, device=device, chunk=chunk, rows=rows)
                      for m, meta in handles], axis=0)
    return preds.mean(axis=0), preds.var(axis=0)


# ---- SMILES-level wrappers ----

def train_regressor(smiles_list, scores, fp_radius=2, fp_nbits=None, fp_type=None, **kwargs):
    fp_nbits = DEFAULT_N_BITS if fp_nbits is None else fp_nbits
    fp_type = DEFAULT_FP_TYPE if fp_type is None else fp_type
    scores = np.asarray(scores, dtype=np.float32)
    kept, X = fingerprint_all(smiles_list, radius=fp_radius, n_bits=fp_nbits, fp_type=fp_type)
    if X.shape[0] == 0:
        raise ValueError("No parseable SMILES in training data.")
    model, meta = train_on_X(X, scores[kept], **kwargs)
    # overwrite the module-default stamp with what ACTUALLY featurized X
    meta.update({"fp_radius": fp_radius, "fp_nbits": fp_nbits, "fp_type": fp_type})
    return model, meta


def predict_scores(model, meta, smiles_list, device=None):
    # A checkpoint written before 2026-08-13 carries no fp_type; it is Morgan by construction, so
    # defaulting to the CURRENT module default would re-featurize it as atom-pair and silently
    # invert its ranking (measured on an equivalent swap: spearman +0.993 -> -0.449, every value
    # finite and in a plausible band, no warning). Absent fp_type therefore means "morgan".
    kept, X = fingerprint_all(smiles_list, radius=meta.get("fp_radius", 2),
                              n_bits=meta.get("fp_nbits", 1024),
                              fp_type=meta.get("fp_type", "morgan"))
    out = np.full(len(smiles_list), np.nan, dtype=np.float32)
    if X.shape[0] > 0:
        pred = predict_on_X(model, meta, X, device=device)
        for local_i, orig_i in enumerate(kept):
            out[orig_i] = pred[local_i]
    return out


def save_regressor(model, meta, out_path):
    """Atomic save (temp file + os.replace) so concurrent readers never see a partial checkpoint."""
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    ckpt = {"state_dict": model.state_dict()}
    ckpt.update(meta)
    tmp = f"{out_path}.tmp-{uuid.uuid4().hex}"
    torch.save(ckpt, tmp)
    os.replace(tmp, out_path)


def load_regressor(model_path, device=None):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model path {model_path} not found.")
    device = get_device(device)
    ckpt = torch.load(model_path, map_location=device)
    model = FNNRegressor(input_dim=ckpt.get("input_dim", 1024)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta
