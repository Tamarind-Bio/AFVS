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
# Provenance: the primary path imports MolPAL's own Featurizer, so "we run MolPAL's featurizer" is a
# fact about the code and not an argument about equivalence. The rdkit fallback exists because the
# head-node AL env is a side-install that a box rebuild loses, and an outage is a worse failure than
# a fallback: it is BIT-IDENTICAL, asserted 200/200 molecules at both widths as a fail-closed gate on
# every Gate 1 run, and which path ran is stamped into the model metadata rather than left ambiguous.
DEFAULT_FP_TYPE = "molpal_pair"
DEFAULT_N_BITS = 2048
_MOLPAL_MAX_LENGTH = 3          # MolPAL Featurizer(fingerprint="pair", radius=2) => minLength 1, maxLength 1+radius

try:
    from molpal.featurizer import Featurizer as _MolpalFeaturizer
    _FEATURIZER_SOURCE = "molpal"
except Exception:                # noqa: BLE001 - any import failure degrades, never breaks a screen
    _MolpalFeaturizer = None
    _FEATURIZER_SOURCE = "rdkit-equivalent"

_molpal_cache = {}


def featurizer_source():
    """Which path produced the fingerprints: 'molpal' or the bit-identical 'rdkit-equivalent'."""
    return _FEATURIZER_SOURCE


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

    if _MolpalFeaturizer is not None:
        key = (radius, n_bits)
        f = _molpal_cache.get(key)
        if f is None:
            f = _molpal_cache[key] = _MolpalFeaturizer(
                fingerprint="pair", radius=radius, length=n_bits)
        v = f(smi)
        return None if v is None else np.asarray(v, dtype=np.float32)

    fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
        mol, minLength=1, maxLength=1 + radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    ConvertToNumpyArray(fp, arr)
    return arr


def fingerprint_all(smiles_list, radius=2, n_bits=None, fp_type=None):
    """
    Fingerprint a list of SMILES, dropping any that fail to parse.
    Returns (kept_indices, X) where kept_indices maps X rows back to smiles_list positions.
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
    # fp_type is stamped from the module default because train_on_X only ever sees a MATRIX, not
    # the SMILES that produced it. A caller that featurized with a non-default fp_type must
    # override it (train_regressor does). Recorded so predict_scores cannot silently re-featurize
    # a checkpoint with a different fingerprint, which inverts the ranking with no error.
    meta = {"input_dim": X.shape[1], "y_mean": y_mean, "y_std": y_std,
            "best_epoch": best_epoch, "best_val_loss": best_val,
            "fp_radius": 2, "fp_nbits": X.shape[1],
            "fp_type": DEFAULT_FP_TYPE, "featurizer_source": _FEATURIZER_SOURCE}
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


def predict_ensemble_on_X(handles, X, device=None, chunk=200_000):
    """Return (mean, var) over an ensemble, both in the ORIGINAL score units.

    var is the population variance of the members' predictions, i.e. how much the ensemble disagrees
    about a ligand. It is an uncertainty SIGNAL for acquisition, not a calibrated posterior, and it
    is documented that way so nobody reads a confidence interval off it. A 1-member ensemble returns
    all-zero variance, which correctly degrades every uncertainty metric back to greedy.
    """
    preds = np.stack([predict_on_X(m, meta, X, device=device, chunk=chunk)
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
