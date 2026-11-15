"""Vectorised paired bootstrap, bit-identical to ER-1's ``paired_bootstrap``.

The random stream matches ER-1 at ``hierarchical=False``: each replicate draws
the in-scope indices and then the OOS indices. AUROC is the same tie-aware
weighted statistic, evaluated for a chunk of replicates at once. Only
``hierarchical=False`` is implemented.
"""

from __future__ import annotations

import numpy as np

from bert_daedl.posthoc.stats import auroc_scores


class FastWeightedAUROC:
    """Tie-aware weighted AUROC, vectorised across multiplicity rows."""

    def __init__(self, id_scores, oos_scores):
        id_scores = np.asarray(id_scores, dtype=np.float64)
        oos_scores = np.asarray(oos_scores, dtype=np.float64)
        self.n_id = int(id_scores.shape[0])
        self.n_oos = int(oos_scores.shape[0])
        scores = np.concatenate([id_scores, oos_scores])
        labels = np.concatenate([
            np.zeros(self.n_id, dtype=np.int8),
            np.ones(self.n_oos, dtype=np.int8),
        ])
        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        self.order = order
        self.lab = labels[order]
        self.starts = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])

    def auroc_many(self, w_id: np.ndarray, w_oos: np.ndarray) -> np.ndarray:
        weights = np.concatenate([w_id, w_oos], axis=1)[:, self.order]
        positive = np.where(self.lab == 1, weights, 0.0)
        negative = weights - positive
        group_pos = np.add.reduceat(positive, self.starts, axis=1)
        group_neg = np.add.reduceat(negative, self.starts, axis=1)
        below = np.cumsum(group_neg, axis=1) - group_neg
        numer = (group_pos * (below + 0.5 * group_neg)).sum(axis=1)
        denom = w_oos.sum(axis=1) * w_id.sum(axis=1)
        return numer / denom


def paired_bootstrap_fast(
    id_a,
    oos_a,
    id_b,
    oos_b,
    *,
    B: int = 5000,
    seed: int = 0,
    hierarchical: bool = False,
    alpha: float = 0.05,
    chunk: int = 250,
) -> dict:
    """Seed-mean of AUROC(A) − AUROC(B) with one shared utterance draw.

    Returns the same fields as ER-1: ``delta``, ``ci_low``, ``ci_high``, ``se``,
    and a two-sided ``p``. ``hierarchical=True`` raises ``NotImplementedError``.
    """
    if hierarchical:
        raise NotImplementedError("paired_bootstrap_fast supports only hierarchical=False")
    if chunk < 1:
        raise ValueError("chunk must be positive")
    id_a = [np.asarray(row, dtype=np.float64) for row in id_a]
    oos_a = [np.asarray(row, dtype=np.float64) for row in oos_a]
    id_b = [np.asarray(row, dtype=np.float64) for row in id_b]
    oos_b = [np.asarray(row, dtype=np.float64) for row in oos_b]
    n_seeds = len(id_a)
    if n_seeds == 0 or len(oos_a) != n_seeds or len(id_b) != n_seeds or len(oos_b) != n_seeds:
        raise ValueError("both arms need the same positive number of seeds")
    n_id = len(id_a[0])
    n_oos = len(oos_a[0])
    for index in range(n_seeds):
        if len(id_a[index]) != n_id or len(id_b[index]) != n_id:
            raise ValueError("in-scope score lengths differ")
        if len(oos_a[index]) != n_oos or len(oos_b[index]) != n_oos:
            raise ValueError("OOS score lengths differ")
    engines_a = [FastWeightedAUROC(id_a[index], oos_a[index]) for index in range(n_seeds)]
    engines_b = [FastWeightedAUROC(id_b[index], oos_b[index]) for index in range(n_seeds)]
    observed = float(np.mean([
        auroc_scores(id_a[index], oos_a[index]) - auroc_scores(id_b[index], oos_b[index])
        for index in range(n_seeds)
    ]))
    rng = np.random.default_rng(seed)
    reps = np.empty(B, dtype=np.float64)
    for start in range(0, B, chunk):
        stop = min(B, start + chunk)
        width = stop - start
        weights_id = np.empty((width, n_id), dtype=np.float64)
        weights_oos = np.empty((width, n_oos), dtype=np.float64)
        for row in range(width):
            weights_id[row] = np.bincount(rng.integers(0, n_id, size=n_id), minlength=n_id)
            weights_oos[row] = np.bincount(rng.integers(0, n_oos, size=n_oos), minlength=n_oos)
        # Mean the per-seed differences in the same order as ER-1's np.mean.
        seed_delta = [
            engines_a[index].auroc_many(weights_id, weights_oos)
            - engines_b[index].auroc_many(weights_id, weights_oos)
            for index in range(n_seeds)
        ]
        reps[start:stop] = np.mean(seed_delta, axis=0)
    low, high = np.quantile(reps, [alpha / 2.0, 1.0 - alpha / 2.0])
    p_le = float(np.mean(reps <= 0))
    p_ge = float(np.mean(reps >= 0))
    p_value = max(2 * min(p_le, p_ge), 1.0 / B)
    return {
        "delta": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "se": float(np.std(reps, ddof=1)) if B > 1 else 0.0,
        "p": float(p_value),
    }
