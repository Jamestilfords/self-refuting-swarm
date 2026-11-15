"""Hand-built FDP, recall, winner's curse, strict truth, and blank CSV cells."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skeptic.analysis.config import AnalysisConfig, definitions_markdown
from skeptic.analysis.hypotheses import run_hypotheses
from skeptic.analysis.report_tables import write_all
from skeptic.analysis.sources import InMemorySource
from skeptic.analysis.truth import (
    FDPResult,
    TruthResult,
    fdp,
    final_pick,
    mean_defined,
    recall,
    true_improvement,
    winners_curse,
)


def _decision(shas, incumbent, cell="cell"):
    rows = []
    current = incumbent
    for sha in shas:
        rows.append({"sha": sha, "incumbent": current, "cell": cell, "accepted": True})
        current = sha
    return pd.DataFrame(rows)


def _flag(candidate, incumbent, cell, primary):
    return TruthResult(
        candidate, incumbent, cell, 0.0, (0.0,), 0.0, 0.0, (0.0, 0.0),
        primary, None, {}, 0, 0,
    )


def test_fdp_arithmetic_and_undefined_zero_accepts():
    truth = {
        ("a", "energy_ce", "cell"): _flag("a", "energy_ce", "cell", True),
        ("b", "a", "cell"): _flag("b", "a", "cell", True),
        ("c", "b", "cell"): _flag("c", "b", "cell", True),
        ("d", "c", "cell"): _flag("d", "c", "cell", False),
        ("e", "d", "cell"): _flag("e", "d", "cell", False),
    }
    five = _decision(["a", "b", "c", "d", "e"], "energy_ce")
    result = fdp(five, truth)
    assert result == FDPResult(5, 2, 0.4)
    empty = fdp(five.iloc[0:0], truth)
    assert empty.fdp is None
    assert empty.n_accepted == 0
    mean, excluded = mean_defined([result.fdp, empty.fdp])
    assert mean == pytest.approx(0.4)
    assert excluded == 1


def test_recall_uses_the_stream_set_and_stays_at_most_one():
    third = recall(pd.DataFrame({"sha": ["a", "d"]}), frozenset({"a", "b", "c"}))
    assert third.recall == pytest.approx(1 / 3)
    assert third.n_true_accepted == 1
    empty = recall(pd.DataFrame({"sha": ["a"]}), frozenset())
    assert empty.recall is None
    # Accepted changes can be true against a weak incumbent and still sit outside S.
    outside = recall(pd.DataFrame({"sha": ["b", "c"]}), frozenset({"a"}))
    assert outside.recall == 0.0
    assert outside.recall <= 1.0


def test_winners_curse_mean_and_change_weighted_h1():
    known = pd.DataFrame({"delta_val": [0.2, 0.5], "delta_test": [0.0, 0.1]})
    curse = winners_curse(known)
    assert curse.n == 2
    assert curse.mean_gap == pytest.approx(0.3)
    assert winners_curse(known.iloc[0:0]).mean_gap is None

    changes = pd.DataFrame([
        {"rule": "R0", "cell": "clinc150/bert", "ordering": 0, "sha": "a",
         "gap": 0.0, "is_false": True, "incumbent": "energy_ce", "delta_val": 0.0, "delta_test": 0.0},
        {"rule": "R0", "cell": "clinc150/bert", "ordering": 1, "sha": "b",
         "gap": 1.0, "is_false": True, "incumbent": "energy_ce", "delta_val": 1.0, "delta_test": 0.0},
        {"rule": "R0", "cell": "clinc150/bert", "ordering": 1, "sha": "c",
         "gap": 1.0, "is_false": True, "incumbent": "b", "delta_val": 1.0, "delta_test": 0.0},
        {"rule": "R0", "cell": "clinc150/bert", "ordering": 1, "sha": "d",
         "gap": 1.0, "is_false": True, "incumbent": "c", "delta_val": 1.0, "delta_test": 0.0},
    ])
    assert winners_curse(changes).mean_gap == pytest.approx(0.75)
    assert changes.groupby("ordering")["gap"].mean().mean() == pytest.approx(0.5)
    config = AnalysisConfig(hyp_B=20)
    report = run_hypotheses(
        changes,
        proposals=["a", "b", "c", "d"],
        truth_sets={cell: frozenset() for cell in config.cells},
        orderings=[0, 1],
        config=config,
    )
    h1 = next(row for row in report["hypotheses"] if row["hypothesis"] == "H1")
    assert h1["estimate"] == pytest.approx(0.75)


def test_strict_reversal_is_only_an_interval_below_zero(monkeypatch):
    from bert_daedl.posthoc.stats import auroc_scores

    def fake(id_a, oos_a, id_b, oos_b, **_kwargs):
        delta = float(np.mean([
            auroc_scores(id_a[index], oos_a[index]) - auroc_scores(id_b[index], oos_b[index])
            for index in range(len(id_a))
        ]))
        return {"delta": delta, "ci_low": delta, "ci_high": delta, "se": 0.0, "p": 1.0}

    monkeypatch.setattr("skeptic.analysis.truth.paired_bootstrap_fast", fake)
    zeros = np.zeros(4)
    ones = np.ones(4)

    def pair(ident, oos):
        return ([ident], [oos])

    data = {
        ("home", "cand"): pair(zeros, ones),
        ("home", "inc"): pair(zeros, zeros),
        ("rev", "cand"): pair(ones, zeros),
        ("rev", "inc"): pair(zeros, ones),
        ("flat", "cand"): pair(zeros, zeros),
        ("flat", "inc"): pair(zeros, zeros),
        ("missing", "cand"): pair(zeros, ones),
    }
    source = InMemorySource("test", data)
    config = AnalysisConfig(
        cells=("home", "rev", "flat", "missing"),
        test_B=10,
        h1_cells=("home",),
        h3_cells=("home",),
        h4_small=("home",),
        h4_large=("home",),
        h2_min_cells=1,
    )
    reversed_result = true_improvement("cand", "inc", "home", True, source=source, config=config)
    assert reversed_result.true_primary is True
    assert reversed_result.true_strict is False
    assert reversed_result.strict_detail["rev"]["reversed"] is True
    assert reversed_result.strict_detail["flat"]["reversed"] is False
    assert "undefined" in reversed_result.strict_detail["missing"]

    no_reversal = AnalysisConfig(
        cells=("home", "flat", "missing"),
        test_B=10,
        h1_cells=("home",),
        h3_cells=("home",),
        h4_small=("home",),
        h4_large=("home",),
        h2_min_cells=1,
    )
    kept = true_improvement("cand", "inc", "home", True, source=source, config=no_reversal)
    assert kept.true_strict is True
    assert "undefined" in kept.strict_detail["missing"]


def test_best_density_is_chosen_on_validation():
    def block(oos_value):
        return ([np.zeros(4)], [np.full(4, oos_value)])

    # On test, d_port looks better. On validation, d_full looks better.
    test = InMemorySource("test", {("cell", name): value for name, value in {
        "energy_ce": block(0.0),
        "kept": block(3.0),
        "d_port_ce": block(3.0),
        "d_full_ce": block(0.0),
    }.items()})
    val = InMemorySource("val", {("cell", name): value for name, value in {
        "energy_ce": block(0.0),
        "kept": block(3.0),
        "d_port_ce": block(0.0),
        "d_full_ce": block(3.0),
    }.items()})
    decisions = pd.DataFrame([
        {"rule": "R0", "cell": "cell", "ordering": 0, "idx": 0, "sha": "kept",
         "incumbent": "energy_ce", "accepted": True},
        {"rule": "R0", "cell": "cell", "ordering": 1, "idx": 0, "sha": "kept",
         "incumbent": "energy_ce", "accepted": False},
    ])
    config = AnalysisConfig(
        cells=("cell",),
        h1_cells=("cell",),
        h3_cells=("cell",),
        h4_small=("cell",),
        h4_large=("cell",),
        h2_min_cells=1,
        density_names=("d_port_ce", "d_full_ce"),
    )
    table = final_pick(decisions, test, config=config, val_source=val)
    row = table.iloc[0]
    assert row["ref_best_density"] == "d_full_ce"
    assert row["ref_best_density_auroc"] == pytest.approx(0.5)
    assert row["ref_best_density_auroc"] < 1.0


def test_definitions_cover_the_registered_metrics():
    text = definitions_markdown()
    for phrase in (
        "False discovery proportion",
        "Recall",
        "Winner's-curse",
        "Final-pick",
        "H1.",
        "H2.",
        "H3.",
        "H4.",
        "Holm",
        "third-smallest",
        "0.5",
        "cannot exceed 1",
    ):
        assert phrase in text


def test_undefined_metrics_are_blank_in_csv(tmp_path):
    cell = "synth/bert"
    config = AnalysisConfig(
        cells=(cell,),
        test_B=10,
        hyp_B=8,
        h1_cells=(cell,),
        h3_cells=(cell,),
        h4_small=(cell,),
        h4_large=(cell,),
        h2_min_cells=1,
        gate_variants=(),
    )

    def block(value):
        return ([np.zeros(4)], [np.full(4, value)])

    payload = {
        (cell, "energy_ce"): block(0.0),
        (cell, "p1"): block(2.0),
        (cell, "p2"): block(0.0),
    }
    test = InMemorySource("test", payload)
    val = InMemorySource("val", payload)
    decisions = pd.DataFrame([
        {"rule": "R0", "cell": cell, "ordering": 0, "idx": 0, "sha": "p1",
         "incumbent": "energy_ce", "accepted": True},
        {"rule": "R0", "cell": cell, "ordering": 0, "idx": 1, "sha": "p2",
         "incumbent": "p1", "accepted": True},
        {"rule": "R0", "cell": cell, "ordering": 1, "idx": 0, "sha": "p1",
         "incumbent": "energy_ce", "accepted": False},
    ])
    ledger = pd.DataFrame([
        {"sha": "p1", "cell": cell, "status": "ok", "stream": "templated", "canary": False},
        {"sha": "p2", "cell": cell, "status": "ok", "stream": "llm", "canary": False},
    ])
    truth = {
        ("p1", "energy_ce", cell): _flag("p1", "energy_ce", cell, False),
        ("p2", "p1", cell): _flag("p2", "p1", cell, True),
    }
    written = write_all(
        tmp_path, decisions=decisions, ledger=ledger, truth=truth,
        test_source=test, val_source=val, config=config, decisions_sha256="abc",
    )
    text = written["exp1_fdp_recall.csv"].read_text(encoding="utf-8")
    assert text.startswith("# config_digest=")
    assert "er1_commit=e877e9159d" in text.splitlines()[0]
    assert "decisions_sha256=abc" in text.splitlines()[0]
    pooled = next(line for line in text.splitlines() if line.startswith("R0,") and ",pooled," in line)
    columns = text.splitlines()[1].split(",")
    values = pooled.split(",")
    recall_at = columns.index("recall_mean")
    assert values[recall_at] == ""
    no_accept = columns.index("n_orderings_no_accept")
    assert values[no_accept] == "1"
