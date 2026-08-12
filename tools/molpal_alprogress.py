"""Translate a MolPAL Explorer's state into the ALProgress map the Tamarind UI renders.

WHY. `ScreeningProgressCard` renders the whole active-learning story off one DynamoDB map:
the per-round timeline, the mode chips, surrogate quality, best top-K, and the terminal stop
reason. Our own controller publishes that map directly. MolPAL's Explorer does not know the map
exists, so adopting MolPAL as the acquisition engine would leave that card dark. This is the
adapter that keeps it lit.

THREE THINGS THAT ARE NOT A RENAME, and each is a place a naive mapping goes quietly wrong:

1. SIGN. MolPAL maximizes: `Explorer.top_k_avg` is a maximize-space number, positive for good
   binders, because our Objective multiplied the raw docking score by c = -1. The card shows
   DOCKING SCORES, where better is more negative, and the demo row reads -12.10. So every score
   crossing this boundary is flipped back. Skip it and the card cheerfully renders +12.10 and the
   timeline appears to improve upward, which no medicinal chemist reading it will believe.

2. SPEARMAN HAS NO MOLPAL EQUIVALENT. Verified at source with a control: zero occurrences of
   "spearman" anywhere in the molpal package. It is OUR held-out surrogate-quality guardrail, and
   it is not decorative -- our controller uses it to detect a surrogate that cannot learn the
   target and to fall back to random exploration (QUALITY_LOW). Under MolPAL that signal has to be
   computed on the side and handed in, or the run loses its only defence against silently
   returning a partial screen on an unlearnable target. `spearman=None` renders as "n/a", so the
   card DEGRADES GRACEFULLY rather than breaking, which is exactly why this is easy to leave
   broken by accident.

3. STOP REASON. `Explorer.completed` folds four distinct conditions into one bool. The card has a
   label per reason and the terminating status is the single most load-bearing field on it, since
   a run that gave up early currently looks identical to one that finished well. Re-derive which
   condition actually fired instead of reporting a generic STOPPED.

MODE. Our controller emits explore/exploit. MolPAL has no such switch: it runs one acquisition
metric throughout. Reporting that metric name (greedy / ucb / ts / ei / pi) is more informative
than inventing an explore/exploit label. Unknown values render grey in the card, which is correct.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# The exact stop tokens ScreeningProgressCard's AL_STOP_LABELS knows.
STATUS_RUNNING = "RUNNING"
STATUS_CONVERGED = "CONVERGED"
STATUS_BUDGET = "BUDGET"
STATUS_EXHAUSTED = "EXHAUSTED"
STATUS_MAX_ROUNDS = "MAX_ROUNDS"


def _neg(v: Optional[float]) -> Optional[float]:
    """maximize-space -> docking score. See SIGN above."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return -f


def derive_status(explorer: Any) -> str:
    """Which of MolPAL's stopping conditions actually fired.

    Order matters and mirrors `Explorer.completed`: it tests iters, then budget, then pool
    exhaustion, and only then convergence. Reporting the wrong one turns a budget exhaustion into
    a false "Converged", which is the reassuring answer and therefore the dangerous one.
    """
    try:
        if getattr(explorer, "iter", 0) > getattr(explorer, "max_iters", float("inf")):
            return STATUS_MAX_ROUNDS
        if len(explorer) >= getattr(explorer, "budget", float("inf")):
            return STATUS_BUDGET
        if getattr(explorer, "exhausted_pool", False):
            return STATUS_EXHAUSTED
        if explorer.completed:
            return STATUS_CONVERGED
    except Exception:
        return STATUS_RUNNING
    return STATUS_RUNNING


def build_alprogress(
    explorer: Any,
    stages: List[Dict[str, Any]],
    *,
    seed_size: Optional[int] = None,
    last_updated: str,
) -> Dict[str, Any]:
    """Assemble the ALProgress map from an Explorer plus the per-round history WE accumulate.

    `stages` is a list of already-translated round dicts (see `make_stage`), because MolPAL keeps
    no per-iteration history: `Explorer` exposes only its CURRENT top_k_avg and score dict, so a
    timeline has to be recorded round by round as the loop runs. Reconstructing it afterwards is
    not possible, which is the single most important thing to know before wiring this in.

    `last_updated` is passed in rather than generated so the caller owns the clock (and so this is
    deterministically testable).
    """
    topks = [s.get("topkMean") for s in stages if s.get("topkMean") is not None]
    # Docking scores: "best" is the MOST NEGATIVE.
    best_topk = min(topks) if topks else None

    spearmans = [s.get("spearman") for s in stages if s.get("spearman") is not None]
    latest_spearman = spearmans[-1] if spearmans else None

    # MolPAL's own convergence quantity: (top_k_avg - moving_average) / moving_average, in
    # maximize space. It is a RATIO, so it does NOT get sign-flipped.
    last_improvement = None
    if len(topks) >= 2 and topks[-2] not in (None, 0):
        last_improvement = abs((topks[-1] - topks[-2]) / topks[-2])

    return {
        "mode": "activelearning",
        "lastUpdated": last_updated,
        "seed": seed_size if seed_size is not None else 0,
        "budget": getattr(explorer, "budget", 0),
        "maxRounds": getattr(explorer, "max_iters", 0),
        "roundsCompleted": len(stages),
        "stages": stages,
        "bestTopkMean": best_topk,
        "latestSpearman": latest_spearman,
        "lastImprovementFrac": last_improvement,
        "overallStatus": derive_status(explorer),
    }


def make_stage(
    round_index: int,
    docked_cumulative: int,
    top_k_avg_maximize_space: Optional[float],
    *,
    acquisition_metric: str = "greedy",
    spearman: Optional[float] = None,
    status: str = "CONTINUE",
) -> Dict[str, Any]:
    """One entry in the ALProgress stages timeline.

    `docked_cumulative` is CUMULATIVE, not per-round: the card takes the LAST stage's value as the
    total docked, so per-round counts would under-report the total by every prior round.

    `spearman` is optional and defaults to None, which the card renders as "n/a". MolPAL supplies
    no such value; pass one in if the caller computes held-out surrogate quality itself.
    """
    return {
        "round": round_index,
        "dockedLigands": docked_cumulative,
        "topkMean": _neg(top_k_avg_maximize_space),
        "spearman": spearman,
        "mode": acquisition_metric,
        "status": status,
    }
