"""Gate ablations keep their own LORD++ state and their own verdicts."""

from skeptic.critic.config import CriticConfig
from skeptic.critic.lord import LordPlusPlus
from skeptic.critic.verdict import verdict
from tests.fixtures.critic.synth import make_cell, small_true, strong


def _decide(candidate, incumbent, baselines, config, skip):
    state = LordPlusPlus(alpha=config.fdr_alpha, w0=config.fdr_w0)
    return verdict(candidate, incumbent, baselines, state, baselines.cell, config=config, skip=skip)


def test_removing_g3_accepts_a_positive_gain_r2_calls_no_effect():
    """A gain inside the neutral band is NO EFFECT. Skipping G3 accepts it when the mean is positive.

    The band is widened here so the interval is NEUTRAL. Under the registered
    0.001 band this same gain is UNDERPOWERED, and skipping G3 still accepts it.
    """
    cell = make_cell(seed=0, n_id=80, n_oos=40)
    config = CriticConfig(bootstrap_B=40, g4_band=1.0)
    candidate = small_true(cell)
    full = _decide(candidate, cell.incumbent, cell.baselines, config, frozenset())
    ablated = _decide(candidate, cell.incumbent, cell.baselines, config, frozenset({"G3"}))
    assert full.verdict.value == "NO EFFECT"
    assert full.gates["G3"] == "fail"
    assert ablated.verdict.value == "ACCEPT"
    assert ablated.gates["G3"] == "skipped"
    assert ablated.delta_val > 0


def test_removing_g1_accepts_a_gain_r2_calls_inert():
    cell = make_cell(seed=0, n_id=80, n_oos=40)
    config = CriticConfig(bootstrap_B=40, g1_flip_thresh=1.0)
    candidate = strong(cell)
    full = _decide(candidate, cell.incumbent, cell.baselines, config, frozenset())
    ablated = _decide(candidate, cell.incumbent, cell.baselines, config, frozenset({"G1"}))
    assert full.verdict.value == "INERT"
    assert full.gates["G1"] == "fail"
    assert ablated.verdict.value == "ACCEPT"
    assert ablated.gates["G1"] == "skipped"
    assert ablated.gates["G3"] == "pass"
