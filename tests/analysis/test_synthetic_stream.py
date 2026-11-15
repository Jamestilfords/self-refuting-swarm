"""The deck's done-when check: planted effects are recovered on a synthetic stream."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from skeptic.analysis.config import AnalysisConfig
from skeptic.analysis.truth import (
    TruthCache,
    compute_truth_table,
    fdp,
    plan_pairs,
    recall,
    stream_truth_set,
)
from tests.fixtures.analysis.replay import replay
from tests.fixtures.analysis.synth_stream import make_stream


def _mean(values):
    defined = [value for value in values if value is not None]
    return float(np.mean(defined))


def _planted_map(group, mu):
    return {
        (row.sha, row.incumbent, row.cell): SimpleNamespace(
            true_primary=(mu[row.sha] - mu[row.incumbent]) > 0.0,
            true_strict=None,
        )
        for row in group.itertuples(index=False)
    }


def _ordering_metrics(accepted, truth, mu, stream_set, planted_true):
    measured_fdp = []
    planted_fdp = []
    measured_recall = []
    planted_recall = []
    for _ordering, group in accepted.groupby("ordering", sort=True):
        measured_fdp.append(fdp(group, truth).fdp)
        planted_fdp.append(fdp(group, _planted_map(group, mu)).fdp)
        measured_recall.append(recall(group, stream_set).recall)
        planted_recall.append(recall(group, planted_true).recall)
    return measured_fdp, planted_fdp, measured_recall, planted_recall


def test_measured_fdp_and_recall_match_the_planted_stream(tmp_path):
    stream = make_stream(seed=0)
    config = AnalysisConfig(cells=(stream.cell,), test_B=1000, test_seed=0)
    decisions = replay(stream.val, stream.shas, stream.cell, n_orderings=50, seed=0)
    ratchet = decisions.loc[decisions["rule"] == "R0"]
    taken = ratchet.loc[ratchet["accepted"].astype(bool)]
    pairs = plan_pairs(taken, stream.ledger, config=config, strict=False)
    started = time.perf_counter()
    table = compute_truth_table(
        pairs, source=stream.test, config=config, cache=TruthCache(tmp_path / "truth"),
    )
    elapsed = time.perf_counter() - started
    agreement = []
    null_false_positives = []
    for (candidate, incumbent, _cell), result in table.items():
        planted = (stream.mu[candidate] - stream.mu[incumbent]) > 0.0
        agreement.append(result.true_primary == planted)
        if not planted:
            null_false_positives.append(bool(result.true_primary))
    stream_set, _details = stream_truth_set(
        stream.cell, stream.ledger, source=stream.test, config=config, cache=TruthCache(tmp_path / "truth"),
    )
    measured_fdp, planted_fdp, measured_recall, planted_recall = _ordering_metrics(
        taken, table, stream.mu, stream_set, stream.planted_true,
    )
    print(
        f"planted_fdp={_mean(planted_fdp):.4f} measured_fdp={_mean(measured_fdp):.4f} "
        f"planted_recall={_mean(planted_recall):.4f} measured_recall={_mean(measured_recall):.4f} "
        f"truth_agreement={float(np.mean(agreement)):.4f} null_fpr={float(np.mean(null_false_positives)):.4f} "
        f"pairs={len(table)} seconds={elapsed:.1f} per_pair={elapsed / max(len(table), 1):.3f}"
    )
    assert float(np.mean(agreement)) >= 0.97
    assert float(np.mean(null_false_positives)) <= 0.05
    assert abs(_mean(measured_fdp) - _mean(planted_fdp)) <= 0.05
    assert abs(_mean(measured_recall) - _mean(planted_recall)) <= 0.05
    assert len(stream_set.symmetric_difference(stream.planted_true)) <= 1
    assert _mean(planted_fdp) >= 0.3


def test_low_power_recall_does_not_exceed_the_planted_recall(tmp_path, capsys):
    stream = make_stream(seed=1, delta_auc=0.005, n_null=40)
    config = AnalysisConfig(cells=(stream.cell,), test_B=1000, test_seed=0)
    decisions = replay(stream.val, stream.shas, stream.cell, n_orderings=20, seed=0)
    ratchet = decisions.loc[decisions["rule"] == "R0"]
    taken = ratchet.loc[ratchet["accepted"].astype(bool)]
    stream_set, _details = stream_truth_set(
        stream.cell, stream.ledger, source=stream.test, config=config, cache=TruthCache(tmp_path / "truth"),
    )
    measured = []
    planted = []
    for _ordering, group in taken.groupby("ordering", sort=True):
        measured.append(recall(group, stream_set).recall)
        planted.append(recall(group, stream.planted_true).recall)
    power = len(stream_set & stream.planted_true) / len(stream.planted_true)
    measured_recall = None if not any(value is not None for value in measured) else _mean(measured)
    planted_recall = _mean(planted)
    print(
        f"low-power truth detection rate: {power:.4f} "
        f"measured_recall={measured_recall} planted_recall={planted_recall:.4f}"
    )
    assert power < 1.0
    if measured_recall is not None:
        assert measured_recall <= planted_recall + 1e-12
