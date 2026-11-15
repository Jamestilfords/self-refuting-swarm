"""Leaving one gate out moves the failure to the next gate."""

from skeptic.critic import CriticConfig, LordPlusPlus, verdict
from tests.fixtures.critic.synth import (
    beats_ceiling,
    make_cell,
    monotone,
    renamed,
    small_true,
    strong_miscal,
)

FAST = CriticConfig(bootstrap_B=200)


def _decide(candidate, cell, skip=frozenset()):
    state = LordPlusPlus()
    decision = verdict(
        candidate,
        cell.incumbent,
        cell.baselines,
        state,
        cell.cell,
        config=FAST,
        skip=skip,
    )
    return decision, state


def test_skip_moves_the_failure():
    cell = make_cell()
    monotone_decision, _ = _decide(monotone(cell), cell, frozenset({"G1"}))
    assert monotone_decision.verdict.value != "INERT"
    assert monotone_decision.gates["G1"] == "skipped"
    assert monotone_decision.verdict.value == "SCORE-MATCHED"

    renamed_decision, _ = _decide(renamed(cell), cell, frozenset({"G2"}))
    assert renamed_decision.verdict.value != "SCORE-MATCHED"

    miscal, _ = _decide(strong_miscal(cell), cell, frozenset({"G5"}))
    assert miscal.verdict.value != "MISCALIBRATES"

    ceiling, _ = _decide(beats_ceiling(cell), cell, frozenset({"G6"}))
    assert ceiling.verdict.value != "BUG"


def test_removing_g3_accepts_small_true_without_an_fdr_step():
    cell = make_cell()
    decision, state = _decide(small_true(cell), cell, frozenset({"G3"}))
    assert decision.verdict.value == "ACCEPT"
    assert decision.gates["G3"] == "skipped"
    assert "G4" not in decision.gates
    assert state.t == 0
