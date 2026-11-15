"""H1–H4 point estimates, shared bootstrap draws, Holm, and the H2/H3 p-value rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from bert_daedl.posthoc.stats import holm

from skeptic.analysis.config import AnalysisConfig
from skeptic.analysis.hypotheses import run_hypotheses
from skeptic.analysis.report_tables import comment_line, render_csv

CELLS = (
    "clinc150/bert",
    "clinc150/roberta",
    "banking77_oos/bert",
    "banking77_oos/roberta",
)
COLUMNS = [
    "hypothesis", "estimate", "ci_low", "ci_high", "p_two_sided", "p_holm", "reject",
    "supported_by_rule", "n_orderings", "n_excluded_orderings", "n_changes", "config_digest",
]


def _changes(false_for_r2: bool):
    rows = []
    for cell in CELLS:
        rows.append({
            "rule": "R0", "cell": cell, "ordering": 0, "sha": "n",
            "is_false": True, "gap": 0.2, "incumbent": "energy_ce",
        })
        rows.append({
            "rule": "R0", "cell": cell, "ordering": 0, "sha": "t",
            "is_false": False, "gap": 0.2, "incumbent": "n",
        })
        rows.append({
            "rule": "R2", "cell": cell, "ordering": 0, "sha": "t",
            "is_false": false_for_r2, "gap": 0.0, "incumbent": "energy_ce",
        })
    return pd.DataFrame(rows)


def _report(changes, hyp_b=40, rng=None):
    config = AnalysisConfig(hyp_B=hyp_b)
    report = run_hypotheses(
        changes,
        proposals=["n", "t"],
        truth_sets={cell: frozenset({"t"}) for cell in CELLS},
        orderings=[0],
        config=config,
        rng=rng,
    )
    return config, report


def _row(report, name):
    return next(row for row in report["hypotheses"] if row["hypothesis"] == name)


def test_point_estimates():
    _config, report = _report(_changes(False), hyp_b=20)
    assert _row(report, "H1")["estimate"] == pytest.approx(0.2)
    assert _row(report, "H2")["estimate"] == pytest.approx(0.5)
    assert _row(report, "H3")["estimate"] == pytest.approx(1.0)
    assert _row(report, "H4")["estimate"] == pytest.approx(0.0)
    assert all(row["estimate"] == pytest.approx(0.5) for row in report["h2_by_cell"])


def test_h2_p_is_the_third_smallest_cell_p():
    _config, report = _report(_changes(False), hyp_b=30)
    cell_p = sorted(row["p_two_sided"] for row in report["h2_by_cell"])
    assert _row(report, "H2")["p_two_sided"] == cell_p[2]


def test_h3_p_is_two_sided_around_one_half():
    config = AnalysisConfig(hyp_B=40)
    truth_sets = {cell: frozenset({"t"}) for cell in config.h3_cells}

    def frame(r2_accepts_truth: bool):
        rows = []
        for cell in config.h3_cells:
            rows.append({
                "rule": "R0", "cell": cell, "ordering": 0, "sha": "t",
                "is_false": False, "gap": 0.0, "incumbent": "energy_ce",
            })
            if r2_accepts_truth:
                rows.append({
                    "rule": "R2", "cell": cell, "ordering": 0, "sha": "t",
                    "is_false": False, "gap": 0.0, "incumbent": "energy_ce",
                })
        return pd.DataFrame(rows)

    common = dict(
        proposals=["t", "n"],
        truth_sets=truth_sets,
        orderings=[0],
        config=config,
    )
    high = run_hypotheses(frame(True), **common)
    low = run_hypotheses(frame(False), **common)
    assert _row(high, "H3")["p_two_sided"] == pytest.approx(1.0 / config.hyp_B)
    assert _row(low, "H3")["p_two_sided"] == pytest.approx(1.0 / config.hyp_B)
    assert _row(high, "H3")["estimate"] >= config.h3_threshold
    assert _row(low, "H3")["estimate"] < config.h3_threshold
    assert _row(high, "H3")["supported_by_rule"] is True
    assert _row(low, "H3")["supported_by_rule"] is False


def test_holm_receives_all_four_p_values():
    config, report = _report(_changes(False), hyp_b=20)
    raw = {row["hypothesis"]: row["p_two_sided"] for row in report["hypotheses"]}
    assert set(raw) == {"H1", "H2", "H3", "H4"}
    adjusted = holm(raw, alpha=config.holm_alpha)
    for row in report["hypotheses"]:
        assert row["p_holm"] == adjusted[row["hypothesis"]]["p_holm"]
        assert row["reject"] == adjusted[row["hypothesis"]]["reject"]


def test_shared_ordering_draws():
    class Spy:
        def __init__(self, seed):
            self._rng = np.random.default_rng(seed)
            self.integers_calls = 0
            self.multi_calls = 0

        def integers(self, *args, **kwargs):
            self.integers_calls += 1
            return self._rng.integers(*args, **kwargs)

        def multinomial(self, *args, **kwargs):
            self.multi_calls += 1
            return self._rng.multinomial(*args, **kwargs)

    spy = Spy(0)
    config, report = _report(_changes(False), hyp_b=25, rng=spy)
    assert report["hypotheses"]
    assert spy.integers_calls == config.hyp_B
    assert spy.multi_calls == config.hyp_B


def test_bootstrap_csv_is_deterministic():
    config = AnalysisConfig(hyp_B=25)
    kwargs = dict(
        proposals=["n", "t"],
        truth_sets={cell: frozenset({"t"}) for cell in CELLS},
        orderings=[0],
        config=config,
    )
    changes = _changes(False)

    def rendered():
        report = run_hypotheses(changes, **kwargs)
        return render_csv(COLUMNS, report["hypotheses"], comment_line(config, "abc"))

    assert rendered() == rendered()


def test_identical_rules_give_h2_zero_and_an_interval_covering_zero():
    rows = []
    for cell in CELLS:
        for rule in ("R0", "R2"):
            rows.append({
                "rule": rule, "cell": cell, "ordering": 0, "sha": "n",
                "is_false": True, "gap": 0.0, "incumbent": "energy_ce",
            })
    config, report = _report(pd.DataFrame(rows), hyp_b=30)
    h2 = _row(report, "H2")
    assert h2["estimate"] == pytest.approx(0.0)
    assert h2["ci_low"] <= 0.0 <= h2["ci_high"]
    assert config.h2_min_cells == 3


def test_h1_h2_and_h4_need_the_holm_rejection():
    config, report = _report(_changes(False), hyp_b=20)
    raw = {row["hypothesis"]: row["p_two_sided"] for row in report["hypotheses"]}
    adjusted = holm(raw, alpha=config.holm_alpha)
    h2_cells = sum(1 for row in report["h2_by_cell"] if row["supported"])
    expected = {
        "H1": _row(report, "H1")["ci_low"] is not None and _row(report, "H1")["ci_low"] > 0,
        "H2": h2_cells >= config.h2_min_cells,
        "H4": _row(report, "H4")["ci_low"] is not None and _row(report, "H4")["ci_low"] > 0,
    }
    for name, criterion in expected.items():
        assert _row(report, name)["supported_by_rule"] is bool(criterion and adjusted[name]["reject"])
    h3 = _row(report, "H3")
    assert h3["supported_by_rule"] is (h3["estimate"] >= config.h3_threshold)
