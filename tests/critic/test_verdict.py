"""Every synthetic helper returns the verdict its construction implies."""

from __future__ import annotations

import json

from skeptic.critic import CriticConfig, LordPlusPlus, Verdict, verdict
from tests.fixtures.critic.synth import (
    beats_ceiling,
    guard,
    make_cell,
    monotone,
    noise,
    renamed,
    small_true,
    strong,
    strong_miscal,
)

FAST = CriticConfig(bootstrap_B=200)
ROW_KEYS = {
    "alpha_t",
    "cell",
    "ci",
    "comparator",
    "delta_val",
    "detail",
    "er1_commit",
    "fdr_t",
    "gates",
    "idx",
    "incumbent",
    "mdd",
    "p",
    "sha",
    "verdict",
}


def _decide(candidate, cell):
    return verdict(
        candidate,
        cell.incumbent,
        cell.baselines,
        LordPlusPlus(alpha=FAST.fdr_alpha, w0=FAST.fdr_w0),
        cell.cell,
        config=FAST,
    )


def _assert_row(decision):
    row = decision.to_row(idx=7)
    assert set(row) == ROW_KEYS
    json.dumps(row, sort_keys=True, allow_nan=False)
    assert row["verdict"] == decision.verdict.value
    assert row["er1_commit"] == "e877e9159d"
    assert row["detail"]["config_digest"] == FAST.digest()
    assert row["idx"] == 7


def test_helpers_match_their_constructions():
    cell = make_cell()
    expected = (
        (monotone(cell), Verdict.INERT, {"G1": "fail"}),
        (renamed(cell), Verdict.SCORE_MATCHED, {"G1": "pass", "G2": "fail"}),
        (noise(cell), Verdict.NO_EFFECT, {"G1": "pass", "G2": "pass", "G3": "fail"}),
        (strong(cell), Verdict.ACCEPT, None),
        (small_true(cell), Verdict.UNDERPOWERED, None),
        (strong_miscal(cell), Verdict.MISCALIBRATES, None),
        (beats_ceiling(cell), Verdict.BUG, None),
        (guard(cell), Verdict.BUG, {"lock": "fail"}),
    )
    for candidate, label, gates in expected:
        decision = _decide(candidate, cell)
        assert decision.verdict is label, (candidate.name, decision.verdict, decision.gates, decision.delta_val)
        if gates is not None:
            assert decision.gates == gates
        _assert_row(decision)
        if label is Verdict.UNDERPOWERED:
            assert decision.mdd > decision.delta_val
        if candidate.name == "canary_noise":
            assert decision.p == 1.0
            assert decision.fdr_t == 1


def test_scorecard_mode_reaches_benefit_instead_of_matching_the_twin():
    import numpy as np

    from skeptic.critic.types import BaselineSet, ScoreSet

    def scores(name, ident, oos, arm="ce"):
        return ScoreSet(name, arm, (np.asarray(ident, dtype=np.float64),), (np.asarray(oos, dtype=np.float64),))

    incumbent = scores("energy_ce", [0, 0, 0, 0], [4, 4, 4, 4])
    twin = scores("d_port_ce", [0, 1, 2, 3], [0.4, 1.2, 1.8, 2.2])
    candidate = scores("d_port_edl", [0, 1, 2, 3], [0.4, 1.2, 1.8, 2.2], arm="edl")
    baselines = BaselineSet(
        cell="clinc150/bert",
        scores={"energy_ce": incumbent, "d_port_ce": twin, "d_port_edl": candidate},
        g2_names=("energy_ce", "d_port_ce", "d_port_edl"),
        plain_ece={"ce": (0.0,), "edl": (0.0,)},
    )
    stream = verdict(candidate, incumbent, baselines, LordPlusPlus(), "clinc150/bert", config=FAST)
    scorecard = verdict(
        candidate, incumbent, baselines, LordPlusPlus(), "clinc150/bert",
        config=FAST, g2_mode="scorecard",
    )
    assert stream.verdict is Verdict.SCORE_MATCHED
    assert scorecard.verdict is Verdict.NO_EFFECT
    assert scorecard.comparator == "energy_ce"
    assert scorecard.detail["g2_mode"] == "scorecard"
    assert scorecard.fdr_t == 1
