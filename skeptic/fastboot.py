"""A fast drop-in for ER-1's stats.paired_bootstrap (non-hierarchical case). Same draws, same numbers.

ER-1 loops over every utterance in Python for each of the B resamples, which takes minutes per pair at
our validation sizes. This computes the same weighted AUROCs with one matrix product. It draws the
resamples with the same generator calls in the same order, so the output matches ER-1 to rounding error
(tests/test_bridge.py checks this).
"""
from __future__ import annotations

import numpy as np


def _mean_credit_gap(id_a, oos_a, id_b, oos_b) -> tuple[np.ndarray, list[float]]:
    """Seed-mean of credit(A) - credit(B) for every (OOS, in-scope) pair, and the per-seed AUROC gaps."""
    total, per_seed = None, []
    for ia, oa, ib, ob in zip(id_a, oos_a, id_b, oos_b):
        ia, oa, ib, ob = (np.asarray(x, np.float64) for x in (ia, oa, ib, ob))
        gap = (np.sign(oa[:, None] - ia[None, :]) - np.sign(ob[:, None] - ib[None, :])) / 2.0
        per_seed.append(float(gap.mean()))
        total = gap if total is None else total + gap
    return total / len(per_seed), per_seed


def paired_bootstrap(id_a, oos_a, id_b, oos_b, *, B: int = 5000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Seed-mean of AUROC(A) - AUROC(B) with one shared utterance draw per resample. Same keys as ER-1."""
    gap, per_seed = _mean_credit_gap(id_a, oos_a, id_b, oos_b)
    n_oos, n_id = gap.shape
    rng = np.random.default_rng(seed)
    w_id, w_oos = np.empty((B, n_id)), np.empty((B, n_oos))
    for b in range(B):                                     # the same two calls per resample as ER-1
        w_id[b] = np.bincount(rng.integers(0, n_id, size=n_id), minlength=n_id)
        w_oos[b] = np.bincount(rng.integers(0, n_oos, size=n_oos), minlength=n_oos)
    reps = np.empty(B)
    for start in range(0, B, 500):
        sl = slice(start, start + 500)
        reps[sl] = ((w_oos[sl] @ gap) * w_id[sl]).sum(1) / (n_oos * n_id)
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    p = max(2 * min(float(np.mean(reps <= 0)), float(np.mean(reps >= 0))), 1.0 / B)
    return {"delta": float(np.mean(per_seed)), "ci_low": float(lo), "ci_high": float(hi),
            "se": float(np.std(reps, ddof=1)) if B > 1 else 0.0, "p": float(p)}
