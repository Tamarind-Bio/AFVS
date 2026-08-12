"""MolPAL Objective backed by AFVS docking.

WHY THIS FILE EXISTS. MolPAL owns the active-learning loop (pool, surrogate, acquisition,
stopping) and ships NO docking engine: its entire integrator contract is one abstract method on
`molpal.objectives.base.Objective` taking SMILES and returning {smiles: score}. Its own
`DockingObjective` delegates to `pyscreener`, which shells out to externally-installed vina-family
binaries and is not even a declared dependency. AFVS already runs docking at scale on AWS Batch
with the engines we support, so the natural seam is: MolPAL decides WHAT to dock, AFVS docks it.

This adapter is that seam. It lets `molpal.Explorer` drive a real AFVS screen.

CONTRACT NOTES, each load-bearing:

- `forward()` must return a score for EVERY input smiles, using None for "failed to score". MolPAL
  treats None as unscored and will not re-acquire it. Silently dropping keys instead corrupts the
  Explorer's accounting, so we always return a full dict.

- MolPAL maximizes. Docking scores are better when MORE NEGATIVE, so the Objective is constructed
  with minimize=True and the base class flips the sign via `self.c`. Getting this backwards does
  not raise, it just makes the surrogate chase the worst compounds, which is the same silent
  inversion the fingerprint identity guard exists to prevent.

- A docking score >= 0 is a FAILED pose (clash / no fit), not a weak binder. Returning it as a real
  value both misleads the surrogate and, downstream, destroys the colour axis on the chemspace
  plot. We map it to None so MolPAL records it as unscored, which is what it is.

DEPENDENCY NOTE. molpal is not on PyPI and its `install_requires` is incomplete: a clean
`pip install` of it exits 0 and then `import molpal` fails, because `molpal/models/__init__.py`
imports `pytorch_lightning`, which is undeclared. Verified 2026-08-12 on python 3.11: installing
`torch` and `pytorch_lightning` alongside it makes the import succeed. Any image or env that wants
this path must install those two explicitly.
"""

from __future__ import annotations

import logging
from typing import Callable, Collection, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

# A qvina02/vina score at or above this is a failed pose rather than a weak binder.
FAILED_POSE_AT_OR_ABOVE = 0.0


def _import_objective_base():
    """Import MolPAL's ABC lazily.

    Kept out of module scope so this file can be imported (and unit-tested) on a box where molpal
    is not installed, which is every box today. molpal/__init__ pulls in explorer -> models ->
    pytorch_lightning, so a module-scope import here would make the whole controller unimportable
    wherever those are missing.
    """
    from molpal.objectives.base import Objective  # noqa: PLC0415

    return Objective


def build_afvs_objective(dock_batch: Callable[[Collection[str]], Dict[str, Optional[float]]]):
    """Return a MolPAL Objective instance that docks through `dock_batch`.

    `dock_batch` takes an iterable of SMILES and returns {smiles: raw_docking_score_or_None}, in
    the raw sign convention where more negative is better. It is the caller's job to make that
    call actually run AFVS (submit the wave, wait, read the score parquets); this adapter only
    owns the MolPAL-facing contract.

    A factory rather than a module-level subclass because the base class cannot be imported at
    module scope, per _import_objective_base.
    """
    Objective = _import_objective_base()

    class AFVSObjective(Objective):
        """Docks through AFVS. Minimizes, so the base class flips the sign for MolPAL."""

        def __init__(self):
            # minimize=True sets self.c = -1: docking scores are better when more negative, and
            # MolPAL always maximizes externally.
            super().__init__(minimize=True)
            self._dock_batch = dock_batch

        def forward(self, xs: Collection[str], *args, **kwargs) -> Dict[str, Optional[float]]:
            smis = list(xs)
            if not smis:
                return {}
            try:
                raw = self._dock_batch(smis)
            except Exception:
                # One failed wave must not kill a multi-day campaign. Report every molecule as
                # unscored and let the Explorer carry on; it handles None by construction.
                logger.exception("AFVS docking wave failed; reporting %d as unscored", len(smis))
                return {smi: None for smi in smis}

            out: Dict[str, Optional[float]] = {}
            n_failed_pose = 0
            for smi in smis:
                v = raw.get(smi)
                if v is None:
                    out[smi] = None
                    continue
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    out[smi] = None
                    continue
                if v != v or v in (float("inf"), float("-inf")):  # NaN / inf
                    out[smi] = None
                    continue
                if v >= FAILED_POSE_AT_OR_ABOVE:
                    # Not a weak binder, a failed pose. None is the honest encoding.
                    n_failed_pose += 1
                    out[smi] = None
                    continue
                out[smi] = self.c * v
            if n_failed_pose:
                logger.info("%d/%d ligands had a non-negative score (failed pose)", n_failed_pose, len(smis))
            return out

    return AFVSObjective()


def scores_from_parquets(
    score_files: Iterable[str],
    manifest_path: str,
) -> Dict[str, Optional[float]]:
    """Build {smiles: best score} from AFVS round parquets plus the ligand-id -> smiles manifest.

    The parquets are keyed by AFVS ligand id and the pool is keyed by SMILES, so the manifest join
    is REQUIRED. Without it nothing matches, every molecule reads as unscored, and MolPAL would
    acquire a fresh batch every round having learned nothing, which looks like a working run.
    Same join, and the same failure mode, as run_scripts/vs_chemspace.py.
    """
    import glob  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    man = pd.read_parquet(manifest_path, columns=["ligand_id", "smiles"])
    lid_to_smi = {str(a): str(b) for a, b in zip(man["ligand_id"], man["smiles"])}

    best: Dict[str, Optional[float]] = {}
    files = [f for pat in score_files for f in sorted(glob.glob(pat, recursive=True))]
    for path in files:
        df = pd.read_parquet(path)
        cols = {c.lower(): c for c in df.columns}
        lig_col = cols.get("ligand") or cols.get("ligand_id") or cols.get("name")
        score_col = cols.get("score_min") or cols.get("score") or cols.get("score_0")
        if not lig_col or not score_col:
            continue
        for lid, val in zip(df[lig_col], df[score_col]):
            smi = lid_to_smi.get(str(lid))
            if smi is None or val is None:
                continue
            v = float(val)
            if v != v:
                continue
            if smi not in best or v < best[smi]:
                best[smi] = v
    return best
