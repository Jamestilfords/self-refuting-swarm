"""G1: exact flip rate, and the ceil(n/2) vote."""

from __future__ import annotations

import numpy as np

from skeptic.critic.config import CriticConfig
from skeptic.critic.g1 import gate_g1
from skeptic.critic.seeds import majority, votes_required
from skeptic.critic.types import ScoreSet


def _scores(name: str, rows: list[tuple[np.ndarray, np.ndarray]]) -> ScoreSet:
    return ScoreSet(
        name=name,
        arm="ce",
        val_id=tuple(ident for ident, _ in rows),
        val_oos=tuple(oos for _, oos in rows),
    )


def _pair(swap: bool) -> tuple[np.ndarray, np.ndarray]:
    ident = np.array([0.0, 1.0])
    oos = np.array([0.5, 3.0]) if swap else np.array([2.0, 3.0])
    return ident, oos


def test_flip_rate_is_one_quarter():
    inc = _scores("inc", [_pair(False)])
    cand = _scores("cand", [_pair(True)])
    result = gate_g1(cand, inc, CriticConfig())
    assert result.detail["flip_rate"] == [0.25]


def test_three_of_five_vote():
    def run(n_inert: int, n: int = 5):
        rows_inc = [_pair(False) for _ in range(n)]
        rows_cand = [_pair(False) if index < n_inert else _pair(True) for index in range(n)]
        return gate_g1(_scores("cand", rows_cand), _scores("inc", rows_inc), CriticConfig())

    assert run(2).outcome == "pass"
    assert run(3).outcome == "fail"
    assert run(3).detail["inert_seeds"] == 3


def test_majority_for_fewer_seeds():
    assert votes_required(4) == 2
    assert votes_required(3) == 2
    assert majority([True, False, False, False]) is False
    assert majority([True, True, False, False]) is True
    assert majority([True, False, False]) is False
    assert majority([True, True, False]) is True

    def run(n_inert: int, n: int):
        rows_inc = [_pair(False) for _ in range(n)]
        rows_cand = [_pair(False) if index < n_inert else _pair(True) for index in range(n)]
        return gate_g1(_scores("cand", rows_cand), _scores("inc", rows_inc), CriticConfig()).outcome

    assert run(1, 4) == "pass"
    assert run(2, 4) == "fail"
    assert run(1, 3) == "pass"
    assert run(2, 3) == "fail"
