"""CSV tables for ``make report``. Undefined metrics are empty cells, never zero.

Descriptive FDP and recall intervals resample orderings. The winner's-curse
interval and the gate ablation use the two-level bootstrap: shared ordering
draws and proposal weights, without replaying the rules.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from skeptic.analysis.ablation import gate_ablation
from skeptic.analysis.config import DEFAULT, ER1_COMMIT, AnalysisConfig
from skeptic.analysis.decisions import accepted, eligible_shas
from skeptic.analysis.hypotheses import draw_two_level, percentile_interval, run_hypotheses
from skeptic.analysis.sources import ScoreSource
from skeptic.analysis.truth import (
    PairKey,
    TruthResult,
    attach_truth,
    final_pick,
    fdp,
    mean_defined,
    recall,
    truth_sets_from_table,
    winners_curse,
)

_STREAMS = (None, "templated", "llm")


def comment_line(config: AnalysisConfig, decisions_sha256: str) -> str:
    return (
        f"# config_digest={config.digest()} er1_commit={ER1_COMMIT} "
        f"decisions_sha256={decisions_sha256}"
    )


def format_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return ""
        if math.isinf(number):
            return "inf" if number > 0 else "-inf"
        return f"{number:.6f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def render_csv(columns: Sequence[str], rows: Sequence[Mapping], comment: str) -> str:
    buffer = io.StringIO(newline="")
    buffer.write(comment.rstrip("\n") + "\n")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(list(columns))
    for row in rows:
        writer.writerow([format_cell(row.get(column)) for column in columns])
    return buffer.getvalue()


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping], comment: str) -> None:
    path.write_text(render_csv(columns, rows, comment), encoding="utf-8")


def _stream_tags(ledger: pd.DataFrame) -> dict[str, str]:
    tags: dict[str, str] = {}
    if "stream" not in ledger.columns or ledger.empty:
        return tags
    for sha, stream in zip(ledger["sha"].astype(str), ledger["stream"].astype(str)):
        tags.setdefault(sha, stream)
    return tags


def _restrict(shas: frozenset[str], tags: dict[str, str], stream: str | None) -> frozenset[str]:
    if stream is None:
        return shas
    return frozenset(sha for sha in shas if tags.get(sha) == stream)


def _filter_stream(rows: pd.DataFrame, tags: dict[str, str], stream: str | None) -> pd.DataFrame:
    if stream is None or rows.empty:
        return rows
    keep = rows["sha"].astype(str).map(tags) == stream
    return rows.loc[keep]


def _strict_counts(rows: pd.DataFrame, truth: Mapping[PairKey, TruthResult]) -> tuple[int | None, float | None]:
    if rows.empty:
        return None, None
    flags = []
    for row in rows.itertuples(index=False):
        result = truth[(str(row.sha), str(row.incumbent), str(row.cell))]
        if result.true_strict is None:
            return None, None
        flags.append(bool(result.true_strict))
    n_false = sum(1 for flag in flags if not flag)
    return n_false, n_false / len(flags)


def _resample_mean_ci(values: list[float | None], config: AnalysisConfig) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = np.random.default_rng(config.hyp_seed)
    reps = np.empty(config.hyp_B, dtype=np.float64)
    width = len(values)
    for draw in range(config.hyp_B):
        chosen = rng.integers(0, width, size=width)
        mean, _excluded = mean_defined(values[int(index)] for index in chosen)
        reps[draw] = np.nan if mean is None else mean
    return percentile_interval(reps)


def _gap_interval(changes: pd.DataFrame, rule: str, cell: str, proposals: list[str], orderings: list[int], config: AnalysisConfig):
    grouped: dict[int, list[tuple[str, float]]] = {}
    if len(changes):
        subset = changes.loc[(changes["rule"] == rule) & (changes["cell"] == cell)]
        for ordering, group in subset.groupby("ordering", sort=True):
            grouped[int(ordering)] = list(zip(group["sha"].astype(str), group["gap"].astype(float)))
    index = {sha: position for position, sha in enumerate(proposals)}
    rng = np.random.default_rng(config.hyp_seed)
    reps = np.empty(config.hyp_B, dtype=np.float64)
    for draw in range(config.hyp_B):
        positions, weights = draw_two_level(rng, len(orderings), len(proposals))
        numerator = 0.0
        denominator = 0.0
        for position in positions:
            for sha, gap in grouped.get(orderings[int(position)], ()):
                weight = float(weights[index[sha]]) if sha in index else 0.0
                numerator += weight * gap
                denominator += weight
        reps[draw] = np.nan if denominator == 0.0 else numerator / denominator
    return percentile_interval(reps)


def _per_ordering(decisions, attached, truth, truth_sets, tags, config):
    records = []
    if decisions.empty:
        return records
    for (rule, cell), group in decisions.groupby(["rule", "cell"], sort=True):
        orderings = sorted(int(value) for value in group["ordering"].unique())
        cell_accepted = attached.loc[(attached["rule"] == rule) & (attached["cell"] == cell)] if len(attached) else attached
        base_set = truth_sets.get(str(cell), frozenset())
        for stream in _STREAMS:
            stream_set = _restrict(base_set, tags, stream)
            stream_rows = _filter_stream(cell_accepted, tags, stream)
            for ordering in orderings:
                rows = stream_rows.loc[stream_rows["ordering"] == ordering] if len(stream_rows) else stream_rows
                primary = fdp(rows, truth, strict=False)
                n_false_strict, fdp_strict = _strict_counts(rows, truth)
                recalled = recall(rows, stream_set)
                curse = winners_curse(rows) if "delta_val" in rows.columns else None
                records.append({
                    "rule": rule,
                    "cell": cell,
                    "stream": "pooled" if stream is None else stream,
                    "ordering": ordering,
                    "n_orderings": len(orderings),
                    "n_accepted": primary.n_accepted,
                    "n_false": primary.n_false,
                    "n_false_strict": n_false_strict,
                    "fdp": primary.fdp,
                    "fdp_strict": fdp_strict,
                    "n_true_accepted": recalled.n_true_accepted,
                    "recall": recalled.recall,
                    "wc_mean": None if curse is None else curse.mean_gap,
                    "true_in_stream": len(stream_set),
                })
    return records


def _aggregate(records: list[dict], config: AnalysisConfig) -> list[dict]:
    rows = []
    frame = pd.DataFrame(records)
    if frame.empty:
        return rows
    for (rule, cell, stream), group in frame.groupby(["rule", "cell", "stream"], sort=True):
        ordered = group.sort_values("ordering")
        fdp_mean, _ = mean_defined(ordered["fdp"].tolist())
        fdp_low, fdp_high = _resample_mean_ci(ordered["fdp"].tolist(), config)
        recall_mean, _ = mean_defined(ordered["recall"].tolist())
        recall_low, recall_high = _resample_mean_ci(ordered["recall"].tolist(), config)
        strict_mean, _ = mean_defined(ordered["fdp_strict"].tolist())
        rows.append({
            "rule": rule,
            "cell": cell,
            "stream": stream,
            "n_orderings": int(ordered["ordering"].nunique()),
            "n_orderings_no_accept": int((ordered["n_accepted"] == 0).sum()),
            "fdp_mean": fdp_mean,
            "fdp_ci_low": fdp_low,
            "fdp_ci_high": fdp_high,
            "fdp_strict_mean": strict_mean,
            "recall_mean": recall_mean,
            "recall_ci_low": recall_low,
            "recall_ci_high": recall_high,
            "accepted_mean": float(ordered["n_accepted"].mean()),
            "true_in_stream": int(ordered["true_in_stream"].iloc[0]),
        })
    return rows


def _winner_rows(attached: pd.DataFrame, proposals, orderings, config: AnalysisConfig) -> list[dict]:
    rows = []
    if attached.empty:
        return rows
    work = attached.copy()
    work["gap"] = work["delta_val"] - work["delta_test"]
    for (rule, cell), group in work.groupby(["rule", "cell"], sort=True):
        low, high = _gap_interval(work, rule, cell, proposals, orderings, config)
        rows.append({
            "rule": rule,
            "cell": cell,
            "n_orderings": len(orderings),
            "n_changes": int(len(group)),
            "wc_mean": float((group["delta_val"] - group["delta_test"]).mean()),
            "ci_low": low,
            "ci_high": high,
            "mean_delta_val": float(group["delta_val"].mean()),
            "mean_delta_test": float(group["delta_test"].mean()),
        })
    return rows


def _truth_rows(truth: Mapping[PairKey, TruthResult], n_orderings: int) -> list[dict]:
    rows = []
    for key in sorted(truth):
        result = truth[key]
        rows.append({
            "candidate": result.candidate,
            "incumbent": result.incumbent,
            "cell": result.cell,
            "n_orderings": n_orderings,
            "delta_test": result.delta_test,
            "seed_deltas": "[" + ",".join(f"{value:.6f}" for value in result.seed_deltas) + "]",
            "se_boot": result.se_boot,
            "se_combined": result.se_combined,
            "ci95_low": result.ci95[0],
            "ci95_high": result.ci95[1],
            "true_primary": result.true_primary,
            "true_strict": result.true_strict,
            "strict_detail": json.dumps(result.strict_detail, sort_keys=True),
            "B": result.B,
            "seed": result.seed,
        })
    return rows


def write_all(
    out_dir: str | Path = "tables",
    *,
    decisions: pd.DataFrame,
    ledger: pd.DataFrame,
    truth: Mapping[PairKey, TruthResult],
    test_source: ScoreSource,
    val_source: ScoreSource,
    config: AnalysisConfig = DEFAULT,
    decisions_sha256: str = "",
    proposals: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Write every Experiment 1 table. Call this once, after unblinding."""
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if decisions.empty:
        raise ValueError("write_all needs the decision log")
    comment = comment_line(config, decisions_sha256)
    tags = _stream_tags(ledger)
    taken = accepted(decisions)
    attached = attach_truth(taken, truth, val_source) if len(taken) else taken
    if len(attached) and "delta_val" in attached.columns:
        attached = attached.copy()
        attached["gap"] = attached["delta_val"] - attached["delta_test"]
        attached["is_false"] = ~attached["true_primary"].astype(bool)
    truth_sets = truth_sets_from_table(truth, ledger, config)
    records = _per_ordering(decisions, attached, truth, truth_sets, tags, config)
    orderings = sorted(int(value) for value in decisions["ordering"].unique())
    if proposals is None:
        names = []
        for cell in config.cells:
            names.extend(eligible_shas(ledger, cell))
        if len(attached):
            names.extend(attached["sha"].astype(str))
        proposals = sorted(set(names))
    proposal_list = list(proposals)
    summary = _aggregate(records, config)
    per_ordering = [
        {key: row[key] for key in (
            "rule", "cell", "ordering", "n_orderings", "n_accepted", "n_false", "n_false_strict",
            "fdp", "fdp_strict", "n_true_accepted", "recall", "wc_mean",
        )}
        for row in records
        if row["stream"] == "pooled"
    ]
    winner = _winner_rows(attached, proposal_list, orderings, config) if len(attached) else []
    ablation = gate_ablation(
        attached if len(attached) else pd.DataFrame(columns=["rule", "cell", "ordering", "sha", "incumbent", "is_false"]),
        proposals=proposal_list,
        orderings=orderings,
        config=config,
    )
    picks = final_pick(decisions, test_source, config=config, val_source=val_source)
    hyp = run_hypotheses(
        attached if len(attached) else pd.DataFrame(
            columns=["rule", "cell", "ordering", "sha", "is_false", "gap", "incumbent"]
        ),
        proposals=proposal_list,
        truth_sets=truth_sets,
        orderings=orderings,
        config=config,
    )
    files = {
        "exp1_fdp_recall.csv": (
            ["rule", "cell", "stream", "n_orderings", "n_orderings_no_accept", "fdp_mean",
             "fdp_ci_low", "fdp_ci_high", "fdp_strict_mean", "recall_mean", "recall_ci_low",
             "recall_ci_high", "accepted_mean", "true_in_stream"],
            summary,
        ),
        "exp1_per_ordering.csv": (
            ["rule", "cell", "ordering", "n_orderings", "n_accepted", "n_false", "n_false_strict",
             "fdp", "fdp_strict", "n_true_accepted", "recall", "wc_mean"],
            per_ordering,
        ),
        "exp1_winners_curse.csv": (
            ["rule", "cell", "n_orderings", "n_changes", "wc_mean", "ci_low", "ci_high",
             "mean_delta_val", "mean_delta_test"],
            winner,
        ),
        "exp1_gate_ablation.csv": (
            ["variant", "cell", "n_orderings", "n_false_mean", "d_n_false_vs_R2", "ci_low",
             "ci_high", "d_fdp_vs_R2", "slipped_mean"],
            ablation.to_dict(orient="records"),
        ),
        "exp1_final_pick.csv": (
            ["rule", "cell", "n_orderings", "auroc_median", "auroc_iqr", "auroc_min", "auroc_max",
             "fpr95_median", "fpr95_iqr", "fpr95_min", "fpr95_max", "ece_median", "ece_iqr",
             "ece_min", "ece_max", "ref_ce_energy_auroc", "ref_best_density", "ref_best_density_auroc"],
            picks.to_dict(orient="records"),
        ),
        "truth_pairs.csv": (
            ["candidate", "incumbent", "cell", "n_orderings", "delta_test", "seed_deltas", "se_boot",
             "se_combined", "ci95_low", "ci95_high", "true_primary", "true_strict", "strict_detail",
             "B", "seed"],
            _truth_rows(truth, len(orderings)),
        ),
        "hypotheses.csv": (
            ["hypothesis", "estimate", "ci_low", "ci_high", "p_one_sided", "p_holm", "reject",
             "supported_by_rule", "n_orderings", "n_excluded_orderings", "n_changes", "config_digest"],
            hyp["hypotheses"],
        ),
        "h2_by_cell.csv": (
            ["cell", "estimate", "ci_low", "ci_high", "p_one_sided", "n_orderings",
             "n_excluded_orderings", "n_changes", "supported", "config_digest"],
            hyp["h2_by_cell"],
        ),
    }
    written = {}
    for name, (columns, rows) in files.items():
        path = destination / name
        write_csv(path, columns, rows, comment)
        written[name] = path
    return written
