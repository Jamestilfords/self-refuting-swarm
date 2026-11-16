import numpy as np

from skeptic.analysis.check_references import arrays_match


def test_float32_rounding_ties_pass_but_real_differences_fail():
    rng = np.random.default_rng(0)
    rows = [rng.normal(size=500) * 1e-7 + 10.0 for _ in range(5)]           # crowded values: float32 creates ties
    stored = np.stack([r.astype(np.float32) for r in rows]).astype(np.float64)
    assert arrays_match(stored, rows)
    wrong = [r.copy() for r in rows]
    wrong[2][[0, 1]] = wrong[2][[1, 0]] + np.array([1.0, -1.0])           # a real reordering
    assert not arrays_match(stored, wrong)
