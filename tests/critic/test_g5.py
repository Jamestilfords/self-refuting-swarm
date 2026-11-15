"""G5 applies only when the proposal returned probabilities."""

from __future__ import annotations

import numpy as np

from skeptic.critic.config import CriticConfig
from skeptic.critic.g5 import gate_g5
from skeptic.critic.types import BaselineSet, ScoreSet


def _candidate(ece):
    row = np.array([0.0, 1.0])
    return ScoreSet(
        name="cand",
        arm="ce",
        val_id=(row,),
        val_oos=(row + 1.0,),
        ece=ece,
    )


def _baselines(plain: float) -> BaselineSet:
    row = np.array([0.0, 1.0])
    energy = ScoreSet(name="energy_ce", arm="ce", val_id=(row,), val_oos=(row + 1.0,))
    return BaselineSet(
        cell="hand/bert",
        scores={"energy_ce": energy},
        g2_names=("energy_ce",),
        plain_ece={"ce": (plain,)},
    )


def test_not_applicable_without_probs():
    result = gate_g5(_candidate(None), _baselines(0.0), CriticConfig())
    assert result.outcome == "pass"
    assert result.detail["applicable"] is False


def test_tolerance_is_strict():
    assert gate_g5(_candidate((0.01,)), _baselines(0.0), CriticConfig()).outcome == "pass"
    assert gate_g5(_candidate((0.0101,)), _baselines(0.0), CriticConfig()).outcome == "fail"
