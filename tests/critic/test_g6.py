"""G6: a grid-angle fusion is under the ceiling; a perfect score is not."""

from __future__ import annotations

import numpy as np
import pytest

from skeptic.critic.config import CriticConfig
from skeptic.critic.er1 import linear_fusion as real_fusion
from skeptic.critic.g6 import gate_g6
from skeptic.critic.types import BaselineSet, ScoreSet


def _parents():
    rng = np.random.default_rng(0)
    first = ScoreSet(
        name="energy_ce",
        arm="ce",
        val_id=(rng.normal(size=30),),
        val_oos=(rng.normal(size=20),),
    )
    second = ScoreSet(
        name="d_full_ce",
        arm="ce",
        val_id=(rng.normal(size=30),),
        val_oos=(rng.normal(size=20),),
    )
    baselines = BaselineSet(
        cell="hand/bert",
        scores={"energy_ce": first, "d_full_ce": second},
        g2_names=("energy_ce", "d_full_ce"),
        plain_ece={"ce": (0.0,)},
    )
    return first, second, baselines


def test_grid_angle_does_not_beat_the_ceiling():
    first, second, baselines = _parents()
    theta = np.linspace(0.0, np.pi, 720, endpoint=False)[17]
    ident = np.cos(theta) * first.val_id[0] + np.sin(theta) * second.val_id[0]
    oos = np.cos(theta) * first.val_oos[0] + np.sin(theta) * second.val_oos[0]
    candidate = ScoreSet(
        name="fused",
        arm="ce",
        val_id=(ident,),
        val_oos=(oos,),
        parents=("energy", "d_full"),
    )
    result = gate_g6(candidate, baselines, CriticConfig())
    assert result.outcome == "pass"
    assert result.detail["excess"][0] <= 1e-12


def test_perfect_separation_fails():
    _, _, baselines = _parents()
    candidate = ScoreSet(
        name="perfect",
        arm="ce",
        val_id=(np.zeros(30),),
        val_oos=(np.ones(20),),
        parents=("energy", "d_full"),
    )
    result = gate_g6(candidate, baselines, CriticConfig())
    assert result.outcome == "fail"
    assert result.detail["excess"][0] > 0.0001


def test_missing_parent_raises():
    _, _, baselines = _parents()
    candidate = ScoreSet(
        name="orphan",
        arm="ce",
        val_id=(np.zeros(30),),
        val_oos=(np.ones(20),),
        parents=("vacuity", "energy"),
    )
    with pytest.raises(ValueError):
        gate_g6(candidate, baselines, CriticConfig())


def test_linear_fusion_sees_only_validation_arrays(monkeypatch):
    first, second, baselines = _parents()
    seen = []

    def wrapped(e_id, e_oos, d_id, d_oos, n_angles=720):
        seen.append((e_id, e_oos, d_id, d_oos))
        assert e_id["val"] is e_id["test"] is first.val_id[0]
        assert e_oos["val"] is e_oos["test"] is first.val_oos[0]
        assert d_id["val"] is d_id["test"] is second.val_id[0]
        assert d_oos["val"] is d_oos["test"] is second.val_oos[0]
        return real_fusion(e_id, e_oos, d_id, d_oos, n_angles=n_angles)

    monkeypatch.setattr("skeptic.critic.g6.linear_fusion", wrapped)
    candidate = ScoreSet(
        name="fused",
        arm="ce",
        val_id=(first.val_id[0],),
        val_oos=(first.val_oos[0],),
        parents=("energy", "d_full"),
    )
    gate_g6(candidate, baselines, CriticConfig())
    assert len(seen) == 1
