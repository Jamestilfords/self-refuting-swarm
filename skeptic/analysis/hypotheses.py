"""H1–H4 point estimates, the two-level bootstrap, and Holm adjustment.

The bootstrap draws ordering indices once per replicate and reuses them for
every rule and cell. Proposal weights are one multinomial draw, shared across
cells. Rules are not replayed.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from bert_daedl.posthoc.stats import holm
from skeptic.analysis.config import DEFAULT, AnalysisConfig


def draw_two_level(rng, n_orderings: int, n_proposals: int) -> tuple[np.ndarray, np.ndarray]:
    """Shared ordering indices, then proposal multiplicity weights."""
    if n_orderings < 1:
        raise ValueError("the bootstrap needs at least one ordering")
    positions = np.asarray(rng.integers(0, n_orderings, size=n_orderings))
    if n_proposals < 1:
        return positions, np.zeros(0, dtype=np.float64)
    probabilities = np.ones(n_proposals, dtype=np.float64)
    probabilities /= probabilities.sum()
    weights = np.asarray(rng.multinomial(n_proposals, probabilities), dtype=np.float64)
    return positions, weights


def percentile_interval(reps: np.ndarray, alpha: float = 0.05) -> tuple[float | None, float | None]:
    array = np.asarray(reps, dtype=np.float64)
    usable = array[~np.isnan(array)]
    if usable.size == 0:
        return None, None
    low, high = np.quantile(usable, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def bootstrap_p(reps: np.ndarray, threshold: float, replicates: int) -> float:
    """Two-sided bootstrap p, floored at 1/B. The value can exceed 1, as in ER-1."""
    array = np.asarray(reps, dtype=np.float64)
    usable = array[~np.isnan(array)]
    if usable.size == 0:
        return 1.0
    p_le = float(np.mean(usable <= threshold))
    p_ge = float(np.mean(usable >= threshold))
    return float(max(2.0 * min(p_le, p_ge), 1.0 / replicates))


def _as_change_index(changes: pd.DataFrame):
    buckets: dict[tuple[str, str, int], list[tuple[str, bool, float, str]]] = defaultdict(list)
    accepted: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for row in changes.itertuples(index=False):
        key = (str(row.rule), str(row.cell), int(row.ordering))
        buckets[key].append((str(row.sha), bool(row.is_false), float(row.gap), str(row.incumbent)))
        accepted[key].add(str(row.sha))
    return buckets, accepted


def _weighted_fdp(rows, weight_of: Mapping[str, float] | None) -> float | None:
    if not rows:
        return None
    numerator = 0.0
    denominator = 0.0
    for sha, is_false, _gap, _incumbent in rows:
        weight = 1.0 if weight_of is None else float(weight_of.get(sha, 0.0))
        denominator += weight
        if is_false:
            numerator += weight
    if denominator == 0.0:
        return None
    return numerator / denominator


def _weighted_n_false(rows, weight_of: Mapping[str, float] | None) -> float:
    total = 0.0
    for sha, is_false, _gap, _incumbent in rows:
        if not is_false:
            continue
        total += 1.0 if weight_of is None else float(weight_of.get(sha, 0.0))
    return total


def _weighted_recall(accepted: set[str], truth_set: frozenset[str], weight_of) -> float | None:
    if len(truth_set) == 0:
        return None
    numerator = 0.0
    denominator = 0.0
    for sha in truth_set:
        weight = 1.0 if weight_of is None else float(weight_of.get(sha, 0.0))
        denominator += weight
        if sha in accepted:
            numerator += weight
    if denominator == 0.0:
        return None
    return numerator / denominator


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _gap(rows, weight_of) -> tuple[float, float]:
    numerator = 0.0
    denominator = 0.0
    for sha, _is_false, gap, _incumbent in rows:
        weight = 1.0 if weight_of is None else float(weight_of.get(sha, 0.0))
        numerator += weight * gap
        denominator += weight
    return numerator, denominator


class _Evaluator:
    def __init__(self, changes, proposals, truth_sets, orderings, config: AnalysisConfig):
        self.config = config
        self.orderings = [int(value) for value in orderings]
        self.proposals = [str(sha) for sha in proposals]
        self.truth_sets = {str(cell): frozenset(shas) for cell, shas in truth_sets.items()}
        self.buckets, self.accepted = _as_change_index(changes)
        self._check_resample()

    def _check_resample(self) -> None:
        if self.config.hyp_resample != "orderings_and_proposal_weights":
            raise NotImplementedError(
                "only the weight approximation is implemented; full replay is an engine run"
            )

    def _weights(self, vector: np.ndarray | None) -> dict[str, float] | None:
        if vector is None:
            return None
        return {sha: float(vector[index]) for index, sha in enumerate(self.proposals)}

    def _selected(self, positions: Sequence[int] | None) -> list[int]:
        if positions is None:
            return list(self.orderings)
        return [self.orderings[int(index)] for index in positions]

    def h1(self, positions, weight_of) -> float | None:
        numerator = 0.0
        denominator = 0.0
        for cell in self.config.h1_cells:
            for ordering in self._selected(positions):
                num, den = _gap(self.buckets.get((self.config.ratchet_rule, cell, ordering), ()), weight_of)
                numerator += num
                denominator += den
        if denominator == 0.0:
            return None
        return numerator / denominator

    def h2_cell(self, cell: str, positions, weight_of) -> tuple[float | None, int]:
        diffs = []
        excluded = 0
        for ordering in self._selected(positions):
            ratchet = _weighted_fdp(
                self.buckets.get((self.config.ratchet_rule, cell, ordering), ()), weight_of,
            )
            critic = _weighted_fdp(
                self.buckets.get((self.config.critic_rule, cell, ordering), ()), weight_of,
            )
            if ratchet is None or critic is None:
                excluded += 1
                continue
            diffs.append(ratchet - critic)
        return _mean(diffs), excluded

    def h3(self, positions, weight_of) -> float | None:
        def mean_recall(rule: str) -> float | None:
            values = []
            for cell in self.config.h3_cells:
                truth_set = self.truth_sets.get(cell, frozenset())
                for ordering in self._selected(positions):
                    value = _weighted_recall(
                        self.accepted.get((rule, cell, ordering), set()),
                        truth_set,
                        weight_of,
                    )
                    if value is not None:
                        values.append(value)
            return _mean(values)

        ratchet = mean_recall(self.config.ratchet_rule)
        critic = mean_recall(self.config.critic_rule)
        if ratchet is None or critic is None or ratchet == 0.0:
            if ratchet == 0.0 and critic not in (None, 0.0):
                return float("inf")
            return None
        return critic / ratchet

    def h4(self, positions, weight_of) -> float | None:
        def group(cells) -> float | None:
            cell_means = []
            for cell in cells:
                values = []
                for ordering in self._selected(positions):
                    value = _weighted_fdp(
                        self.buckets.get((self.config.ratchet_rule, cell, ordering), ()),
                        weight_of,
                    )
                    if value is not None:
                        values.append(value)
                if values:
                    cell_means.append(float(np.mean(values)))
            return _mean(cell_means)

        small = group(self.config.h4_small)
        large = group(self.config.h4_large)
        if small is None or large is None:
            return None
        return small - large

    def evaluate(self, positions, weights: np.ndarray | None) -> dict:
        weight_of = self._weights(weights)
        per_cell = {}
        for cell in self.config.cells:
            estimate, excluded = self.h2_cell(cell, positions, weight_of)
            per_cell[cell] = {"estimate": estimate, "n_excluded": excluded}
        return {
            "H1": self.h1(positions, weight_of),
            "H2_cells": per_cell,
            "H3": self.h3(positions, weight_of),
            "H4": self.h4(positions, weight_of),
        }


def _order_statistic(values: list[float], k_largest: int) -> float | None:
    if len(values) < k_largest:
        return None
    return float(sorted(values, reverse=True)[k_largest - 1])


def _supported_positive(low: float | None) -> bool:
    return low is not None and low > 0.0


def _with_holm(criterion: bool, name: str, adjusted: dict) -> bool:
    return bool(criterion and adjusted[name]["reject"])


def run_hypotheses(
    changes: pd.DataFrame,
    *,
    proposals: Sequence[str],
    truth_sets: Mapping[str, frozenset[str]],
    orderings: Sequence[int],
    config: AnalysisConfig = DEFAULT,
    rng=None,
) -> dict:
    """Point estimates and two-level intervals for H1–H4."""
    required = {"rule", "cell", "ordering", "sha", "is_false", "gap", "incumbent"}
    missing = required.difference(changes.columns)
    if missing and not changes.empty:
        raise ValueError(f"changes are missing {sorted(missing)}")
    if changes.empty:
        changes = pd.DataFrame(columns=list(required))
    evaluator = _Evaluator(changes, proposals, truth_sets, orderings, config)
    observed = evaluator.evaluate(None, None)
    replicates = config.hyp_B
    cell_reps = {cell: np.full(replicates, np.nan) for cell in config.cells}
    stat_reps = {name: np.full(replicates, np.nan) for name in ("H1", "H2", "H3", "H4")}
    generator = rng if rng is not None else np.random.default_rng(config.hyp_seed)
    for draw in range(replicates):
        positions, weights = draw_two_level(generator, len(evaluator.orderings), len(evaluator.proposals))
        sample = evaluator.evaluate(positions, weights)
        for cell in config.cells:
            value = sample["H2_cells"][cell]["estimate"]
            cell_reps[cell][draw] = np.nan if value is None else value
        finite_cells = [
            cell_reps[cell][draw] for cell in config.cells if np.isfinite(cell_reps[cell][draw])
        ]
        if len(finite_cells) >= config.h2_min_cells:
            stat_reps["H2"][draw] = float(sorted(finite_cells, reverse=True)[config.h2_min_cells - 1])
        for name in ("H1", "H3", "H4"):
            value = sample[name]
            stat_reps[name][draw] = np.nan if value is None else value
    cell_rows = []
    cell_p = {}
    for cell in config.cells:
        estimate = observed["H2_cells"][cell]["estimate"]
        low, high = percentile_interval(cell_reps[cell])
        p_value = bootstrap_p(cell_reps[cell], 0.0, replicates)
        cell_p[cell] = p_value
        n_changes = int(((changes["cell"] == cell) & changes["rule"].isin(
            [config.ratchet_rule, config.critic_rule]
        )).sum()) if len(changes) else 0
        cell_rows.append({
            "cell": cell,
            "estimate": estimate,
            "ci_low": low,
            "ci_high": high,
            "p_two_sided": p_value,
            "n_orderings": len(evaluator.orderings),
            "n_excluded_orderings": int(observed["H2_cells"][cell]["n_excluded"]),
            "n_changes": n_changes,
            "supported": _supported_positive(low),
            "config_digest": config.digest(),
        })
    if len(config.cells) < config.h2_min_cells:
        raise ValueError("h2_min_cells exceeds the number of cells")
    ordered_p = sorted(cell_p[cell] for cell in config.cells)
    h2_p = float(ordered_p[config.h2_min_cells - 1])
    h1_p = bootstrap_p(stat_reps["H1"], 0.0, replicates)
    h3_p = bootstrap_p(stat_reps["H3"], config.h3_threshold, replicates)
    h4_p = bootstrap_p(stat_reps["H4"], 0.0, replicates)
    raw = {"H1": h1_p, "H2": h2_p, "H3": h3_p, "H4": h4_p}
    adjusted = holm(raw, alpha=config.holm_alpha)
    h1_low, h1_high = percentile_interval(stat_reps["H1"])
    h2_low, h2_high = percentile_interval(stat_reps["H2"])
    h3_low, h3_high = percentile_interval(stat_reps["H3"])
    h4_low, h4_high = percentile_interval(stat_reps["H4"])
    h2_cells_clear = sum(1 for row in cell_rows if row["supported"])
    h1_changes = changes.loc[
        (changes["rule"] == config.ratchet_rule) & changes["cell"].isin(config.h1_cells)
    ] if len(changes) else changes
    h4_cells = set(config.h4_small) | set(config.h4_large)
    h4_changes = changes.loc[
        (changes["rule"] == config.ratchet_rule) & changes["cell"].isin(h4_cells)
    ] if len(changes) else changes
    h2_excluded = sum(row["n_excluded_orderings"] for row in cell_rows)
    h3_defined = observed["H3"]
    rows = [
        _row("H1", observed["H1"], h1_low, h1_high, h1_p, adjusted,
             _with_holm(_supported_positive(h1_low), "H1", adjusted),
             len(evaluator.orderings), 0, int(len(h1_changes)), config),
        _row("H2", _h2_estimate(observed, config), h2_low, h2_high, h2_p, adjusted,
             _with_holm(h2_cells_clear >= config.h2_min_cells, "H2", adjusted),
             len(evaluator.orderings), h2_excluded, int(len(changes)), config),
        _row("H3", h3_defined, h3_low, h3_high, h3_p, adjusted,
             h3_defined is not None and h3_defined >= config.h3_threshold,
             len(evaluator.orderings), 0, int(len(changes)), config),
        _row("H4", observed["H4"], h4_low, h4_high, h4_p, adjusted,
             _with_holm(_supported_positive(h4_low), "H4", adjusted),
             len(evaluator.orderings), 0, int(len(h4_changes)), config),
    ]
    return {"hypotheses": rows, "h2_by_cell": cell_rows}


def _h2_estimate(observed: dict, config: AnalysisConfig) -> float | None:
    values = []
    for cell in config.cells:
        estimate = observed["H2_cells"][cell]["estimate"]
        if estimate is not None and np.isfinite(estimate):
            values.append(float(estimate))
    return _order_statistic(values, config.h2_min_cells)


def _row(name, estimate, low, high, p_value, adjusted, supported, n_orderings, n_excluded, n_changes, config):
    return {
        "hypothesis": name,
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "p_two_sided": p_value,
        "p_holm": adjusted[name]["p_holm"],
        "reject": adjusted[name]["reject"],
        "supported_by_rule": supported,
        "n_orderings": n_orderings,
        "n_excluded_orderings": n_excluded,
        "n_changes": n_changes,
        "config_digest": config.digest(),
    }
