"""Bit-identical paired bootstrap against ER-1, including ties and seed counts."""

from __future__ import annotations

import numpy as np
import pytest
from bert_daedl.posthoc.stats import paired_bootstrap

from skeptic.analysis.bootstrap import paired_bootstrap_fast

_FIELDS = ("delta", "ci_low", "ci_high", "se", "p")


def _arms(n_id, n_oos, n_seeds, seed, decimals=None):
    rng = np.random.default_rng(seed)
    arms = {"id_a": [], "oos_a": [], "id_b": [], "oos_b": []}
    for _ in range(n_seeds):
        draws = {
            "id_a": rng.normal(0.0, 1.0, size=n_id),
            "oos_a": rng.normal(1.0, 1.0, size=n_oos),
            "id_b": rng.normal(0.1, 1.0, size=n_id),
            "oos_b": rng.normal(0.7, 1.0, size=n_oos),
        }
        if decimals is not None:
            draws = {key: np.round(value, decimals) for key, value in draws.items()}
        for key, value in draws.items():
            arms[key].append(value)
    return arms


CASES = [
    ("clinc_val", 3000, 100, 2, 1, None),
    ("clinc_test", 4500, 1000, 2, 2, None),
    ("banking", 2000, 2080, 2, 3, None),
    ("heavy_ties", 400, 200, 3, 4, 2),
    ("one_seed", 300, 80, 1, 5, None),
    ("five_seeds", 300, 80, 5, 6, None),
]


@pytest.mark.parametrize(("name", "n_id", "n_oos", "n_seeds", "seed", "decimals"), CASES)
def test_matches_er1(name, n_id, n_oos, n_seeds, seed, decimals):
    arms = _arms(n_id, n_oos, n_seeds, seed, decimals)
    fast = paired_bootstrap_fast(**arms, B=20, seed=0, hierarchical=False)
    slow = paired_bootstrap(**arms, B=20, seed=0, hierarchical=False)
    for field in _FIELDS:
        assert fast[field] == slow[field], (name, field, fast[field], slow[field])


def test_hierarchical_is_rejected():
    arms = _arms(20, 10, 2, 0)
    with pytest.raises(NotImplementedError):
        paired_bootstrap_fast(**arms, B=4, hierarchical=True)


def test_chunk_size_does_not_change_draws():
    arms = _arms(80, 40, 3, 7, decimals=2)
    whole = paired_bootstrap_fast(**arms, B=30, seed=11, chunk=30)
    pieces = paired_bootstrap_fast(**arms, B=30, seed=11, chunk=7)
    for field in _FIELDS:
        assert whole[field] == pieces[field]
