#!/usr/bin/env python3
"""Equivalence guard for the CHUNKED acquisition path in afvs_al_select.select_ligands.

There is no pytest surface in this repo, so this is a runnable script.

WHAT IT GUARDS. `select_ligands` used to build four full-pool structures around an acquirer that
already streams: a Python list of N ligand-id strings, two N-wide float arrays, and an N-entry
position dict, plus two more copies and the utility vector inside acquire_batch. At a 1e9 pool that
is the dominant memory ask of the round, and it is paid to select a few thousand ligands. The pool
is now fed to that same acquirer in chunks, so those structures are bounded by one chunk.

The risk in that change is not a crash. It is a SILENTLY DIFFERENT SELECTION: a chunked pass that
kept a slightly worse set, or the right set in a different order, produces a screen that runs to
completion and reports healthy numbers while docking the wrong molecules. Nothing downstream raises.

So the reference here is the SHIPPED single-pass branch of the same function, reached by making the
chunk larger than the pool, not a reimplementation of the old code. The two branches are compared on
ligand_id ORDER, on the raw `pred` bytes, on collection_key, and on the column set.

Every assertion carries a control that must FAIL. Three of those controls are the point of the file
rather than decoration:
  - CHUNKING ACTUALLY HAPPENED. A differential test between two paths is vacuous if both ran the
    same code, and the cheapest way for this one to go vacuous is a chunk size that silently exceeds
    the pool. Every comparison therefore asserts the acquirer pass COUNT, which is 1 for the
    reference and several for the chunked run. This is the same hazard as a parallel test that
    passes at max_workers=1.
  - THE RUNNING FOLD IS LOAD-BEARING. The obvious wrong implementation lets each chunk's top-B
    overwrite the last instead of carrying it forward. That is emulated here and asserted to DIFFER
    from the reference, so if the shipped code ever stops folding, the equivalence assertion above
    is known to catch it. The pass COUNT is asserted alongside, because folding and
    accumulate-then-reduce give the SAME answer and differ only in memory: one pass per chunk means
    folding, one per chunk plus a final reduction means accumulating.
  - THE STOCHASTIC EXCLUSION IS LOAD-BEARING. CHUNKABLE_METRICS excludes random/ts/noisy/threshold.
    This measures the reason directly rather than asserting the exclusion list back to itself: two
    same-seeded acquirers are shown to draw IDENTICAL random utilities, which is what would turn a
    chunked random round into a positional bias wearing the costume of a random sample.

    python tools/verify_select_streaming.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "vendor"))

import afvs_al_select as S  # noqa: E402
from molpal.acquirer.acquirer import Acquirer  # noqa: E402

# ---- acquirer-pass spy. The pass COUNT is what separates a real chunked run from a vacuous one. ----
_ORIG_ACQUIRE = Acquirer.acquire_batch
PASSES = {"n": 0, "sizes": []}


def _spy(self, xs, *args, **kwargs):
    PASSES["n"] += 1
    PASSES["sizes"].append(len(xs))
    return _ORIG_ACQUIRE(self, xs, *args, **kwargs)


Acquirer.acquire_batch = _spy

FAILURES = []


def check(ok, label, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        FAILURES.append(label)
    return ok


def make_pool(n, seed=0):
    """A synthetic pool whose ligand-id order is UNCORRELATED with row position.

    The tie-break inside the acquirer's heap is the ligand-id string, so an id scheme that happens
    to sort the same way as the rows would hide a reordering bug. These ids are a shuffled zero-
    padded integer space, and `pred` carries deliberate duplicate values so the tie path is live.
    """
    rng = np.random.RandomState(seed)
    ids = [f"lig{i:08d}" for i in rng.permutation(n)]
    pred = np.round(rng.normal(-8.0, 1.5, size=n), 3)  # rounding manufactures real ties in `pred`
    return pd.DataFrame({
        "collection_key": [f"AA_{i // 97:07d}" for i in range(n)],
        "ligand_id": ids,
        "smiles": [f"C{'C' * (i % 11)}O" for i in range(n)],
        "pred": pred.astype(np.float32),
    })


def run(pool, budget, metric, chunk_rows, docked=(), docked_scores=None, pred_var=None,
        explore=False, seed=42):
    """Call the SHIPPED select_ligands with a given chunk size; return (frame, n_acquirer_passes)."""
    prev = os.environ.get("alSelectChunkRows")
    os.environ["alSelectChunkRows"] = str(chunk_rows)
    PASSES["n"], PASSES["sizes"] = 0, []
    try:
        out = S.select_ligands(pool, list(docked), budget, metric=metric, explore=explore,
                               seed=seed, pred_var=pred_var, docked_scores=docked_scores)
    finally:
        if prev is None:
            os.environ.pop("alSelectChunkRows", None)
        else:
            os.environ["alSelectChunkRows"] = prev
    return out, PASSES["n"]


def same(a, b):
    """Compare two selections the way the caller consumes them: order, ids, pred BYTES, columns.

    The INDEX is compared too. That is not pedantry: select_ligands returns rows whose index is
    their POSITION in the pool, which is what normalising a caller's odd index buys, and it is the
    only observable difference an unconditional skip of that normalisation produces now that
    everything else in the function addresses rows positionally.
    """
    if list(a.columns) != list(b.columns):
        return False, f"columns {list(a.columns)} vs {list(b.columns)}"
    if list(a.ligand_id.astype(str)) != list(b.ligand_id.astype(str)):
        return False, f"ligand_id order/set differs ({len(a)} vs {len(b)} rows)"
    if a.pred.to_numpy().tobytes() != b.pred.to_numpy().tobytes():
        return False, "pred bytes differ"
    if list(a.collection_key.astype(str)) != list(b.collection_key.astype(str)):
        return False, "collection_key differs"
    if list(a.index) != list(b.index):
        return False, f"index differs ({list(a.index)[:4]} vs {list(b.index)[:4]})"
    return True, f"{len(a)} rows identical"


# ============================ 1. equivalence across metric x budget ============================
print("\n1. CHUNKED == SINGLE PASS, over every chunkable metric and several budgets")
N = 1200
POOL = make_pool(N, seed=0)
# A docked set exercises BOTH things acquire_batch reads `explored` for: the exclusion test and
# `current_max`, which ei and pi consume as the incumbent. Without real scores current_max is -inf
# and both metrics degenerate to a total tie, which would make this whole section far weaker.
DOCKED = list(POOL.ligand_id[:40].astype(str))
DOCKED_SCORES = {lid: float(v) for lid, v in zip(DOCKED, POOL.pred[:40].astype(float))}
RNG = np.random.RandomState(7)
PRED_VAR = np.abs(RNG.normal(0.5, 0.25, size=N))

n_compared = 0
for metric in sorted(S.CHUNKABLE_METRICS):
    for budget in (1, 7, 50):
        for var_label, pv in (("var=None", None), ("var=real", PRED_VAR)):
            ref, ref_passes = run(POOL, budget, metric, chunk_rows=10 * N,
                                  docked=DOCKED, docked_scores=DOCKED_SCORES, pred_var=pv)
            chk, chk_passes = run(POOL, budget, metric, chunk_rows=100,
                                  docked=DOCKED, docked_scores=DOCKED_SCORES, pred_var=pv)
            ok, detail = same(ref, chk)
            label = f"{metric:>6s} budget={budget:<3d} {var_label}"
            # The two pass-count asserts are what stop this from being a comparison of one path
            # with itself. select_chunk_rows floors the chunk at 2*budget, so at budget=50 the
            # effective chunk is 100 and the pool still splits into 12 of them.
            check(ref_passes == 1, f"{label}: reference is ONE pass", f"passes={ref_passes}")
            check(chk_passes >= 4, f"{label}: chunked really chunked", f"passes={chk_passes}")
            check(ok, f"{label}: selections identical", detail)
            n_compared += 1
print(f"  ({n_compared} metric x budget x variance combinations compared)")

# The chunked path carries a RUNNING top-B into the next chunk's pass, rather than accumulating
# every chunk's picks and reducing at the end. Both give the same answer, so no comparison above can
# tell them apart, but only the folding one is O(chunk + budget): accumulating holds
# pool * (budget/chunk) rows, which at the 2x-budget floor is half the pool. The fold is visible
# here as the pass COUNT: exactly one per chunk, with no separate reduction pass at the end.
_, fold_passes = run(POOL, 7, "greedy", chunk_rows=100, docked=DOCKED, docked_scores=DOCKED_SCORES)
n_chunks = -(-N // 100)
check(fold_passes == n_chunks,
      "one acquirer pass PER CHUNK: the running best is folded in, not accumulated",
      f"{fold_passes} passes over {n_chunks} chunks")

# SPARSE POOL. On a dense pool every chunk fills the budget, so the running best is always full and
# every pass reduces a budget-plus-chunk candidate set. The sparse case is the opposite shape: each
# chunk contributes a handful and the running set never reaches the budget at all, so the fold is
# carrying a partial answer forward the whole way. It is the case where an implementation that
# forgot to carry `best` into the next pass still looks plausible on a dense pool.
SPARSE_DOCKED, SPARSE_KEEP = [], 3
for _start in range(0, N, 100):
    _blk = list(POOL.ligand_id[_start:_start + 100].astype(str))
    SPARSE_DOCKED.extend(_blk[SPARSE_KEEP:])
SPARSE_SCORES = {lid: float(v) for lid, v in
                 zip(SPARSE_DOCKED, POOL.pred[:len(SPARSE_DOCKED)].astype(float))}
_undocked = N - len(SPARSE_DOCKED)
s_ref, s_ref_p = run(POOL, 50, "greedy", chunk_rows=10 * N,
                     docked=SPARSE_DOCKED, docked_scores=SPARSE_SCORES)
s_chk, s_chk_p = run(POOL, 50, "greedy", chunk_rows=100,
                     docked=SPARSE_DOCKED, docked_scores=SPARSE_SCORES)
ok, detail = same(s_ref, s_chk)
check(len(s_ref) == _undocked and _undocked < 50,
      "CONTROL sparse pool never fills the budget, so the fold never trims",
      f"{_undocked} undocked vs budget 50")
check(s_chk_p == n_chunks, "CONTROL sparse run really used the chunked path",
      f"{s_chk_p} passes == {n_chunks} chunks")
check(ok, "sparse pool: chunked still matches the single pass", detail)

# ============================ 2. control: the comparator can fail ============================
print("\n2. CONTROL: the comparator detects a wrong selection (else section 1 proves nothing)")
ref, _ = run(POOL, 7, "greedy", chunk_rows=10 * N, docked=DOCKED, docked_scores=DOCKED_SCORES)
ok, detail = same(ref, ref.iloc[::-1])
check(not ok, "reversed selection is REJECTED", detail)
ok, detail = same(ref, ref.drop(columns=["smiles"]))
check(not ok, "selection missing the smiles column is REJECTED", detail)
perturbed = ref.copy()
perturbed.loc[perturbed.index[0], "pred"] = np.float32(-999.0)
ok, detail = same(ref, perturbed)
check(not ok, "one changed pred byte is REJECTED", detail)

# ============================ 3. control: the final pass is load-bearing ============================
print("\n3. CONTROL: dropping the final pass over the union CHANGES the answer")
# Emulate the obvious wrong implementation: concatenate the per-chunk top-B and return it. Each
# chunk is selected with a single-pass call, which is exactly what the chunk loop does.
CHUNK = 100
naive_differs = 0
for metric in ("greedy", "pi"):
    ref, _ = run(POOL, 7, metric, chunk_rows=10 * N, docked=DOCKED, docked_scores=DOCKED_SCORES)
    parts = []
    for start in range(0, N, CHUNK):
        part = POOL.iloc[start:start + CHUNK]
        sel, _ = run(part, 7, metric, chunk_rows=10 * N, docked=DOCKED, docked_scores=DOCKED_SCORES)
        if len(sel):
            parts.append(sel)
    naive = pd.concat(parts)
    ok, _ = same(ref, naive)
    if not ok:
        naive_differs += 1
    check(not ok, f"{metric}: no-final-pass concatenation differs from the reference",
          f"{len(naive)} candidate rows vs {len(ref)} selected")
check(naive_differs == 2, "the final pass is load-bearing for BOTH metrics tested")

# ============================ 4. control: pred_var actually reaches the metric ============================
print("\n4. CONTROL: pred_var is positional and load-bearing (else its alignment is untested)")
a, _ = run(POOL, 20, "ucb", chunk_rows=100, docked=DOCKED, docked_scores=DOCKED_SCORES,
           pred_var=PRED_VAR)
b, _ = run(POOL, 20, "ucb", chunk_rows=100, docked=DOCKED, docked_scores=DOCKED_SCORES,
           pred_var=np.roll(PRED_VAR, 1))
ok, _ = same(a, b)
check(not ok, "ucb with a SHIFTED pred_var selects differently", "so the alignment is real")
c, _ = run(POOL, 20, "ucb", chunk_rows=100, docked=DOCKED, docked_scores=DOCKED_SCORES,
           pred_var=None)
ok, _ = same(a, c)
check(not ok, "ucb with variance differs from ucb without", "so zeros are not silently substituted")

try:
    run(POOL, 20, "ucb", chunk_rows=100, pred_var=PRED_VAR[:-1])
    check(False, "a short pred_var RAISES")
except ValueError as e:
    check("positional" in str(e), "a short pred_var RAISES", str(e)[:60])

# ============================ 5. stochastic metrics are excluded, and why ============================
print("\n5. STOCHASTIC metrics take the single pass, and the reason is measured not asserted")
for metric in ("random", "ts", "thompson", "noisy", "threshold"):
    _, passes = run(POOL, 7, metric, chunk_rows=100, docked=DOCKED, docked_scores=DOCKED_SCORES)
    check(passes == 1, f"{metric:>9s}: one pass even at chunk=100", f"passes={passes}")
_, passes = run(POOL, 7, "greedy", chunk_rows=100, docked=DOCKED, docked_scores=DOCKED_SCORES,
                explore=True)
check(passes == 1, "explore=True forces random, so ONE pass", f"passes={passes}")
# The mechanism itself: same seed -> same draw. Two acquirers over two DIFFERENT chunks of equal
# length select the same POSITIONS, which is the positional bias the exclusion exists to prevent.
left, right = make_pool(200, seed=1), make_pool(200, seed=2)
picks = []
for frame in (left, right):
    acq = Acquirer(size=200, init_size=5, batch_sizes=[5], metric="random", epsilon=0.0,
                   seed=42, verbose=0)
    xs = list(frame.ligand_id.astype(str))
    chosen = acq.acquire_batch(xs=xs, y_means=-frame.pred.to_numpy(dtype=float),
                               y_vars=np.zeros(200), explored={}, t=1)
    picks.append(sorted(xs.index(c) for c in chosen))
check(picks[0] == picks[1],
      "two same-seeded random acquirers pick IDENTICAL POSITIONS in different chunks",
      f"{picks[0]} == {picks[1]}")

# ============================ 6. select_chunk_rows ============================
print("\n6. select_chunk_rows: default, override, floor, and a bad value")
os.environ.pop("alSelectChunkRows", None)
check(S.select_chunk_rows(10) == S.DEFAULT_SELECT_CHUNK_ROWS, "unset -> default")
check(S.select_chunk_rows(10 ** 9) == 2 * 10 ** 9, "floors at 2x budget so chunking still prunes")
# The non-finite rows are not padding. int(float("inf")) raises OverflowError, which is NOT a
# subclass of ValueError, so an except tuple that lists only (TypeError, ValueError) lets it escape
# and kills the round at init. "-inf" is literally the non-positive case the docstring promises to
# warn about and fall back from, so it is a direct self-contradiction rather than an edge case.
DEF = S.DEFAULT_SELECT_CHUNK_ROWS
for raw, expect in (("5000", 5000), ("  5000  ", 5000), ("1e6", 1_000_000),
                    ("None", DEF), ("", DEF), ("null", DEF), ("banana", DEF),
                    ("-4", DEF), ("0", DEF), ("nan", DEF),
                    ("inf", DEF), ("-inf", DEF), ("1e400", DEF), ("Infinity", DEF)):
    os.environ["alSelectChunkRows"] = raw
    try:
        got, err = S.select_chunk_rows(1), None
    except Exception as exc:                    # a raise here is a FAIL, not a crash of the guard
        got, err = None, f"{type(exc).__name__}: {exc}"
    check(got == expect, f"alSelectChunkRows={raw!r} -> {expect:,}",
          err or f"got {got:,}")

# The WARN must not reach STDOUT: run-virtual-screening.sh captures this process's stdout and greps
# it for the controller's STATUS line, so a caller-supplied string echoed there is on a parsed
# stream. Capturing stdout is the only way to see this; the message looks identical either way.
import contextlib                                                        # noqa: E402
import io                                                                # noqa: E402
os.environ["alSelectChunkRows"] = "banana STATUS=CONVERGED"
_out, _err = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(_out), contextlib.redirect_stderr(_err):
    S.select_chunk_rows(1)
check(_out.getvalue() == "", "a bad value writes NOTHING to stdout",
      f"stdout was {_out.getvalue()!r}")
check("STATUS=CONVERGED" in _err.getvalue(),
      "CONTROL the warning really was emitted (on stderr), so the check above is not vacuous")
os.environ.pop("alSelectChunkRows", None)

# ============================ 7. edges ============================
print("\n7. EDGES: empty pool, zero budget, fully-docked pool, budget past the pool")
empty, _ = run(POOL.iloc[0:0], 5, "greedy", chunk_rows=100)
check(len(empty) == 0 and list(empty.columns) == list(POOL.columns), "empty pool -> empty, columns kept")
zero, _ = run(POOL, 0, "greedy", chunk_rows=100)
check(len(zero) == 0, "budget 0 -> empty")
alldocked, _ = run(POOL, 7, "greedy", chunk_rows=100, docked=list(POOL.ligand_id.astype(str)))
check(len(alldocked) == 0, "fully-docked pool -> empty (EXHAUSTED upstream)")
big_ref, _ = run(POOL, N * 2, "greedy", chunk_rows=10 * N, docked=DOCKED)
big_chk, chk_passes = run(POOL, N * 2, "greedy", chunk_rows=100, docked=DOCKED)
ok, detail = same(big_ref, big_chk)
check(ok, "budget larger than the pool: identical", detail)
check(chk_passes == 1, "budget > pool cannot chunk (2x-budget floor exceeds the pool)",
      f"passes={chk_passes}")

# ============================ 8. duplicate ligand ids ============================
print("\n8. DUPLICATE ligand_ids: the fold must stay associative when an id resolves to >1 row")
# THIS IS THE SECTION THAT MATTERS MOST, and an earlier version of it was VACUOUS BY CONSTRUCTION.
# It duplicated the pool with `pd.concat([make_pool(300, seed=3), make_pool(300, seed=3)])`, the
# same seed twice, so every duplicated ligand_id had a byte-identical twin in pred as well as id.
# Equal preds mean equal utilities, so first-occurrence and best-occurrence resolve to the same row
# and the hazard cannot appear. The duplicates that matter carry DIFFERENT preds, which is exactly
# what build_manifest produces: it joins on a key with `\.\d+$` stripped from both sides, designed
# to collide tautomer and protomer suffixes, so one ligand_id fans out into rows with different
# smiles and therefore different predictions.
dup = pd.concat([make_pool(300, seed=3), make_pool(300, seed=4)], ignore_index=True)
dup["ligand_id"] = list(make_pool(300, seed=3).ligand_id) * 2      # ids collide, preds do not
n_dup_diff = int((dup.groupby("ligand_id").pred.nunique() > 1).sum())
check(n_dup_diff > 200, "CONTROL the fixture really carries duplicated ids with DIFFERING preds",
      f"{n_dup_diff} of {dup.ligand_id.nunique()} ids have >1 distinct pred")
d_ref, _ = run(dup, 9, "greedy", chunk_rows=10 * len(dup))
d_chk, d_passes = run(dup, 9, "greedy", chunk_rows=100)
ok, detail = same(d_ref, d_chk)
check(ok, "duplicated pool: identical", detail)
check(d_passes >= 4, "duplicated pool really chunked", f"passes={d_passes}")

# The minimal regression case, reduced from the divergence a review found. Pre-fix, single-pass
# returned ligand A and the chunked path returned an entirely different ligand C, because the row
# carried forward for A was its -1.0 occurrence rather than the -100.0 one that won the heap slot.
MIN = pd.DataFrame({"collection_key": ["K"] * 6,
                    "ligand_id": ["A", "A", "B", "C", "D", "E"],
                    "smiles": ["C"] * 6,
                    "pred": np.array([-1.0, -100.0, -9.0, -8.0, -7.0, -6.0], dtype=np.float32)})
m_ref, _ = run(MIN, 1, "greedy", chunk_rows=1000)
m_chk, m_p = run(MIN, 1, "greedy", chunk_rows=3)
ok, detail = same(m_ref, m_chk)
check(m_p > 1, "CONTROL the minimal case really chunked", f"passes={m_p}")
check(ok, "minimal duplicate-id case: chunked == single pass", detail)
check(float(m_ref.pred.iloc[0]) == -100.0,
      "a duplicated id resolves to its BEST-predicted row, not whichever came first",
      f"pred={float(m_ref.pred.iloc[0])}")

# ======================= 9. the RangeIndex fast path is INDEX-AGNOSTIC =======================
print("\n9. NON-RangeIndex callers: same selection, and the caller's frame is not mutated")
# select_ligands skips reset_index when the index is already RangeIndex(0, n, 1), which is a 22
# B/molecule copy avoided. That is only safe if a caller whose index is NOT that shape still gets
# the same answer, and if aliasing the caller's frame never writes through to it.
#
# THIS SECTION MUST RUN WITH pred_var, and the reason is the finding that a mutation control turned
# up. With pred_var=None the index is never read at all: `_pick` returns POSITIONS and every
# consumer below is .iloc, so an unconditional skip of reset_index passes an index-agnostic test
# perfectly. The single place the index is load-bearing is `pv[frame.index.to_numpy()]`. So the
# var=None rows below are the weak half and are kept only to show the paths agree; the var=real
# rows are what actually guards the fast path.
base_pool = make_pool(900, seed=11)
PRED_VAR9 = np.abs(np.random.RandomState(9).normal(0.5, 0.25, size=900))
odd_indexes = {
    "shuffled unique": np.random.RandomState(5).permutation(900),
    "non-zero-start range": pd.RangeIndex(start=500, stop=1400),
    "descending step": pd.RangeIndex(start=899, stop=-1, step=-1),
    "string labels": [f"k{i}" for i in range(900)],
    "non-unique": [i // 2 for i in range(900)],
}
for var_label, pv9, metric9 in (("var=None", None, "greedy"), ("var=real", PRED_VAR9, "ucb")):
    ref9, _ = run(base_pool, 11, metric9, chunk_rows=100, docked=DOCKED,
                  docked_scores=DOCKED_SCORES, pred_var=pv9)
    for label, index in odd_indexes.items():
        odd = base_pool.copy()
        odd.index = index
        before = odd.copy(deep=True)
        try:
            got, passes = run(odd, 11, metric9, chunk_rows=100, docked=DOCKED,
                              docked_scores=DOCKED_SCORES, pred_var=pv9)
            ok, detail = same(ref9, got)
        except Exception as exc:            # a non-positional index lookup raises; that is a FAIL
            ok, passes, detail = False, -1, f"{type(exc).__name__}: {str(exc)[:50]}"
        check(ok, f"{var_label} index={label}: same selection as a RangeIndex pool", detail)
        check(passes >= 4 or not ok, f"{var_label} index={label}: really chunked", f"passes={passes}")
        check(before.equals(odd), f"{var_label} index={label}: caller's frame unmutated")
# The fast path must actually be exercised, or section 1 measured the reset_index branch throughout
# and the skip is untested. A RangeIndex pool must come back sharing the caller's data.
plain = make_pool(900, seed=11)
plain_before = plain.copy(deep=True)
_ = run(plain, 11, "greedy", chunk_rows=100, docked=DOCKED, docked_scores=DOCKED_SCORES)
check(plain_before.equals(plain), "RangeIndex pool: caller's frame unmutated")
check(isinstance(plain.index, pd.RangeIndex) and plain.index.start == 0 and plain.index.step == 1,
      "CONTROL the section-1 pool really is RangeIndex(0, n, 1)", "so the fast path was exercised")

# ============================ verdict ============================
print("\n" + "=" * 78)
if FAILURES:
    print(f"FAILED ({len(FAILURES)}):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("ALL CHECKS PASSED: the chunked acquisition path is selection-identical to the single pass,")
print("every comparison was proven non-vacuous by the acquirer pass count, and the two exclusions")
print("(carrying the running best forward, excluding the stochastic metrics) each change the answer.")
