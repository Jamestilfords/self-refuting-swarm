"""The four deck canaries, at the critic rather than through the sandbox."""

from skeptic.critic import CriticConfig, LordPlusPlus, verdict
from tests.fixtures.critic.synth import guard, make_cell, monotone, noise, renamed

FAST = CriticConfig(bootstrap_B=200)


def _verdict(candidate, cell):
    return verdict(
        candidate,
        cell.incumbent,
        cell.baselines,
        LordPlusPlus(),
        cell.cell,
        config=FAST,
    ).verdict.value


def test_four_canaries():
    cell = make_cell()
    assert _verdict(monotone(cell), cell) == "INERT"
    assert _verdict(renamed(cell), cell) == "SCORE-MATCHED"
    assert _verdict(noise(cell), cell) == "NO EFFECT"
    assert _verdict(guard(cell), cell) == "BUG"
