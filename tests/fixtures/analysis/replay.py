"""Minimal reference ratchet and a toy critic, both on validation AUROC.

The ratchet accepts when the seed-mean validation AUROC is strictly higher
than the incumbent's. The toy critic accepts only when that gain exceeds
2.5 times its seed-level standard error. Both rules see the same permutations
from ``default_rng(0)`` and write the E5 decision fields, including ``accepted``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bert_daedl.posthoc.stats import auroc_scores
from skeptic.analysis.decisions import write_decision_tree
from skeptic.analysis.sources import ScoreSource


def _seed_aurocs(source: ScoreSource, cell: str, name: str) -> np.ndarray:
    ident, oos = source.get(cell, name)
    return np.asarray(
        [auroc_scores(ident[index], oos[index]) for index in range(len(ident))],
        dtype=np.float64,
    )


def _ratchet(delta: float, _seeds: np.ndarray) -> tuple[bool, str]:
    take = delta > 0.0
    return take, "ACCEPT" if take else "REJECT"


def _critic(delta: float, seeds: np.ndarray) -> tuple[bool, str]:
    if len(seeds) < 2:
        scale = 0.0
    else:
        scale = float(np.std(seeds, ddof=1) / np.sqrt(len(seeds)))
    take = delta > 2.5 * scale
    return take, "ACCEPT" if take else "NO EFFECT"


def replay(
    val: ScoreSource,
    shas: list[str],
    cell: str,
    *,
    n_orderings: int = 50,
    seed: int = 0,
    reference: str = "energy_ce",
    root: str | Path | None = None,
) -> pd.DataFrame:
    """Matched R0 and toy-R2 decision rows for one cell."""
    rng = np.random.default_rng(seed)
    permutations = [rng.permutation(len(shas)) for _ in range(n_orderings)]
    auroc = {reference: _seed_aurocs(val, cell, reference)}
    for sha in shas:
        auroc[sha] = _seed_aurocs(val, cell, sha)
    rows = []
    for rule, decide in (("R0", _ratchet), ("R2", _critic)):
        for ordering, perm in enumerate(permutations):
            incumbent = reference
            for idx, position in enumerate(perm):
                sha = shas[int(position)]
                per_seed = auroc[sha] - auroc[incumbent]
                delta = float(per_seed.mean())
                take, verdict = decide(delta, per_seed)
                rows.append({
                    "rule": rule,
                    "ordering": ordering,
                    "sha": sha,
                    "cell": cell,
                    "idx": idx,
                    "incumbent": incumbent,
                    "accepted": take,
                    "verdict": verdict,
                    "delta_val": delta,
                    "comparator": incumbent,
                })
                if take:
                    incumbent = sha
    frame = pd.DataFrame(rows)
    if root is not None:
        write_decision_tree(frame, root)
    return frame
