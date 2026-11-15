"""LORD++ golden levels from the onlineFDR gamma sequence."""

from __future__ import annotations

import pytest

from skeptic.critic.lord import LordPlusPlus, gamma

GOLDEN_GAMMA = {
    1: 0.053516770913,
    2: 0.011638205783,
    3: 0.009912498794,
    10: 0.003898251905,
    100: 0.000415842546,
}

# t, p, alpha_t, reject
GOLDEN_PATH = (
    (1, 0.001, 0.002675838546, True),
    (2, 0.5, 0.003257748835, False),
    (3, 0.0005, 0.001077535229, True),
    (4, 1.0, 0.006259482334, False),
    (5, 0.002, 0.001925444367, False),
    (6, 0.9, 0.001642988382, False),
)


def test_gamma_constants():
    for index, expected in GOLDEN_GAMMA.items():
        assert gamma(index) == pytest.approx(expected, abs=1e-12)


def test_golden_path_and_boundary():
    state = LordPlusPlus()
    rejections = []
    for step, p, alpha_t, reject in GOLDEN_PATH:
        assert state.level() == pytest.approx(alpha_t, abs=1e-12)
        assert state.level() == pytest.approx(alpha_t, abs=1e-12)
        assert state.t == step - 1
        assert state.test(p) is reject
        if reject:
            rejections.append(step)
        assert state.rejections == tuple(rejections)
    assert state.rejections == (1, 3)

    fresh = LordPlusPlus()
    alpha_1 = fresh.level()
    assert fresh.test(alpha_1) is True


def test_round_trip_and_copy_continue():
    state = LordPlusPlus()
    for _, p, _, _ in GOLDEN_PATH[:3]:
        state.test(p)
    restored = LordPlusPlus.from_dict(state.to_dict())
    cloned = state.copy()
    assert restored.level() == state.level()
    assert cloned.test(0.9) == state.test(0.9)
    assert restored.to_dict()["t"] == 3
    assert cloned.rejections == state.rejections


def test_wealth_constraints():
    with pytest.raises(ValueError):
        LordPlusPlus(alpha=0.1, w0=0.2)
    with pytest.raises(ValueError):
        LordPlusPlus(alpha=1.0, w0=0.05)
    with pytest.raises(ValueError):
        LordPlusPlus(alpha=0.1, w0=0.0)
    total = 0.0
    for index in range(1, 10_001):
        total += gamma(index)
        assert 0.0 < total < 1.0
