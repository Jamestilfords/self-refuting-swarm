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
