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
from rdkit.Chem import AllChem, rdFingerprintGenerator
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


# Which fingerprint the surrogate featurizes with. Changing this changes what every model
# trained afterwards means, which is why it is recorded in the checkpoint and checked on load.
#
# "atompair" is what MolPAL used for BOTH its RF and its NN. A controlled ablation on the
# archived recall harness (our own network, everything else held fixed, three seeds, with a
# shuffled-label negative control) moved recall of the true top-1% at 6% docked from 0.581 to
# 0.723 by changing this alone. Bit width contributed nothing (-0.017), so 1024 stays.
#
# Caveat kept deliberately visible: that ablation is ONE target (4UNN), a retrospective lookup
# against published AutoDock Vina scores. Prior work established that recall transfers from Vina
# to our qvina02 (0.75 vs 0.77), but the atom-pair win itself has not been re-measured on our
# engine. The AL loop retrains every round behind a held-out Spearman gate, so a featurizer that
# does not help shows up as low Spearman and trips explore-then-exploit rather than silently
# degrading a screen.
DEFAULT_FP_TYPE = "morgan"

_VALID_FP_TYPES = ("morgan", "atompair")


def smi_to_fingerprint(smi, radius=2, n_bits=1024, fp_type=None):
    """SMILES -> binary fingerprint bit vector, or None if unparseable/empty.

    fp_type "morgan" (radius 2) or "atompair". Atom-pair encodes topological distance between
    atom-type pairs rather than circular environments, so it separates actives that share a
    scaffold but differ in substitution pattern, which is where the recall difference shows up.
    """
    if not smi or smi in ("N/A", "None", "nan"):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    ftype = fp_type or DEFAULT_FP_TYPE
    if ftype == "atompair":
        # rdFingerprintGenerator is the non-deprecated path and returns the same ExplicitBitVect
        # the Morgan call does, so the GetOnBits loop below is unchanged.
        gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
        fp = gen.GetFingerprint(mol)
    elif ftype == "morgan":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    else:
        raise ValueError(f"unknown fp_type {ftype!r}; expected one of {_VALID_FP_TYPES}")
    arr = np.zeros((n_bits,), dtype=np.float32)
    for bit in fp.GetOnBits():
        arr[bit] = 1.0
    return arr


def fingerprint_all(smiles_list, radius=2, n_bits=1024, fp_type=None):
    """
    Fingerprint a list of SMILES, dropping any that fail to parse.
    Returns (kept_indices, X) where kept_indices maps X rows back to smiles_list positions.
    """
    kept_indices, fps = [], []
    for i, smi in enumerate(smiles_list):
        fp = smi_to_fingerprint(smi, radius=radius, n_bits=n_bits, fp_type=fp_type)
        if fp is not None:
            kept_indices.append(i)
            fps.append(fp)
    if not fps:
        return [], np.zeros((0, n_bits), dtype=np.float32)
    return kept_indices, np.stack(fps, axis=0)


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
    # fp_type is what a later load checks against. train_on_X never sees the featurizer, so it
    # records the module default; train_regressor overrides both when it featurized explicitly.
    # (fp_radius was previously a hardcoded 2 here regardless of what produced X.)
    meta = {"input_dim": X.shape[1], "y_mean": y_mean, "y_std": y_std,
            "best_epoch": best_epoch, "best_val_loss": best_val,
            "fp_type": DEFAULT_FP_TYPE, "fp_nbits": X.shape[1]}
    if DEFAULT_FP_TYPE == "morgan":
        meta["fp_radius"] = 2
    return model, meta


def predict_on_X(model, meta, X, device=None, chunk=200_000):
    """Predict docking scores (original units) for a precomputed fingerprint matrix X, chunked."""
    device = get_device(device)
    model = model.to(device)
    model.eval()
    X = np.asarray(X, dtype=np.float32)
    out = np.empty(X.shape[0], dtype=np.float32)
    with torch.no_grad():
        for start in range(0, X.shape[0], chunk):
            xb = torch.from_numpy(X[start:start + chunk]).to(device)
            z = model(xb).squeeze(1).cpu().numpy()
            out[start:start + chunk] = z * meta["y_std"] + meta["y_mean"]
    return out


# ---- SMILES-level wrappers ----

def train_regressor(smiles_list, scores, fp_radius=2, fp_nbits=1024, fp_type=None, **kwargs):
    scores = np.asarray(scores, dtype=np.float32)
    ftype = fp_type or DEFAULT_FP_TYPE
    kept, X = fingerprint_all(smiles_list, radius=fp_radius, n_bits=fp_nbits, fp_type=ftype)
    if X.shape[0] == 0:
        raise ValueError("No parseable SMILES in training data.")
    model, meta = train_on_X(X, scores[kept], **kwargs)
    meta.update({"fp_type": ftype, "fp_nbits": fp_nbits})
    # radius is meaningless for atom-pair, so do not carry a misleading value.
    if ftype == "morgan":
        meta["fp_radius"] = fp_radius
    else:
        meta.pop("fp_radius", None)
    return model, meta


def predict_scores(model, meta, smiles_list, device=None):
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
    # A checkpoint trained on one fingerprint and used with another does NOT raise on its own:
    # both are the same width, so load_state_dict cannot object, predictions come back finite and
    # inside the plausible kcal/mol band, and the ranking is INVERTED (measured Spearman +0.993
    # to -0.449). An acquisition ranked on that preferentially docks the worst ligands. A missing
    # fp_type means the checkpoint predates this field, i.e. it was written by Morgan-only code.
    ckpt_fp = meta.get("fp_type", "morgan")
    if ckpt_fp != DEFAULT_FP_TYPE:
        raise ValueError(
            f"{model_path} was trained with fp_type={ckpt_fp!r} but this code featurizes with "
            f"{DEFAULT_FP_TYPE!r}. Predictions would be silently mis-ranked, not merely noisy. "
            f"Retrain the surrogate, or pin DEFAULT_FP_TYPE to {ckpt_fp!r}."
        )
    return model, meta
