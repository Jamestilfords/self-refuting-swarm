"""Leave-one-gate-out false discoveries against the full critic.

G4 has no variant: it only relabels a proposal. Slipped changes are false
accepts in a variant whose candidate and incumbent the full critic did not
accept on that same ordering.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from skeptic.analysis.config import DEFAULT, AnalysisConfig
from skeptic.analysis.hypotheses import draw_two_level, percentile_interval
from skeptic.analysis.truth import TruthResult, fdp


def _index(changes: pd.DataFrame):
    buckets = defaultdict(list)
    for row in changes.itertuples(index=False):
        key = (str(row.rule), str(row.cell), int(row.ordering))
        buckets[key].append((str(row.sha), str(row.incumbent), bool(row.is_false)))
    return buckets


def _n_false(rows) -> int:
    return sum(1 for _sha, _inc, is_false in rows if is_false)


def _weighted_n_false(rows, weight_of: dict[str, float]) -> float:
    return sum(weight_of.get(sha, 0.0) for sha, _inc, is_false in rows if is_false)


def _slipped(variant_rows, full_rows) -> int:
    full_pairs = {(sha, incumbent) for sha, incumbent, _is_false in full_rows}
    return sum(
        1
        for sha, incumbent, is_false in variant_rows
        if is_false and (sha, incumbent) not in full_pairs
    )


def gate_ablation(
    changes: pd.DataFrame,
    *,
    proposals: list[str],
    orderings: list[int],
    config: AnalysisConfig = DEFAULT,
    rng=None,
) -> pd.DataFrame:
    """Mean false discoveries of each variant, and the paired gap versus R2."""
    if config.hyp_resample != "orderings_and_proposal_weights":
        raise NotImplementedError("gate intervals use the same weight approximation as H1–H4")
    buckets = _index(changes) if len(changes) else {}
    population = [int(value) for value in orderings]
    proposal_ids = [str(sha) for sha in proposals]
    generator = rng if rng is not None else np.random.default_rng(config.hyp_seed)
    rows = []
    for variant in config.gate_variants:
        for cell in config.cells:
            diffs = []
            fdp_diffs = []
            slipped = []
            false_counts = []
            for ordering in population:
                variant_rows = buckets.get((variant, cell, ordering), [])
                full_rows = buckets.get((config.critic_rule, cell, ordering), [])
                variant_false = _n_false(variant_rows)
                full_false = _n_false(full_rows)
                false_counts.append(variant_false)
                diffs.append(variant_false - full_false)
                slipped.append(_slipped(variant_rows, full_rows))
                variant_frame = _frame(variant, cell, ordering, variant_rows)
                full_frame = _frame(config.critic_rule, cell, ordering, full_rows)
                variant_fdp = fdp(variant_frame, _truth_of(variant_rows, cell)).fdp
                full_fdp = fdp(full_frame, _truth_of(full_rows, cell)).fdp
                if variant_fdp is not None and full_fdp is not None:
                    fdp_diffs.append(variant_fdp - full_fdp)
            reps = np.full(config.hyp_B, np.nan)
            for draw in range(config.hyp_B):
                positions, weights = draw_two_level(generator, len(population), len(proposal_ids))
                weight_of = {sha: float(weights[index]) for index, sha in enumerate(proposal_ids)}
                chosen = [population[int(index)] for index in positions]
                paired = []
                for ordering in chosen:
                    variant_rows = buckets.get((variant, cell, ordering), [])
                    full_rows = buckets.get((config.critic_rule, cell, ordering), [])
                    paired.append(
                        _weighted_n_false(variant_rows, weight_of) - _weighted_n_false(full_rows, weight_of)
                    )
                reps[draw] = float(np.mean(paired)) if paired else np.nan
            low, high = percentile_interval(reps)
            rows.append({
                "variant": variant,
                "cell": cell,
                "n_orderings": len(population),
                "n_false_mean": float(np.mean(false_counts)) if false_counts else None,
                "d_n_false_vs_R2": float(np.mean(diffs)) if diffs else None,
                "ci_low": low,
                "ci_high": high,
                "d_fdp_vs_R2": float(np.mean(fdp_diffs)) if fdp_diffs else None,
                "slipped_mean": float(np.mean(slipped)) if slipped else None,
            })
    return pd.DataFrame(rows)


def _frame(rule: str, cell: str, ordering: int, rows) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["rule", "cell", "ordering", "sha", "incumbent"])
    return pd.DataFrame([
        {"rule": rule, "cell": cell, "ordering": ordering, "sha": sha, "incumbent": incumbent}
        for sha, incumbent, _is_false in rows
    ])


def _truth_of(rows, cell: str) -> dict[tuple[str, str, str], TruthResult]:
    truth = {}
    for sha, incumbent, is_false in rows:
        truth[(sha, incumbent, cell)] = TruthResult(
            candidate=sha,
            incumbent=incumbent,
            cell=cell,
            delta_test=0.0,
            seed_deltas=(),
            se_boot=0.0,
            se_combined=0.0,
            ci95=(0.0, 0.0),
            true_primary=not is_false,
            true_strict=None,
            strict_detail={},
            B=0,
            seed=0,
        )
    return truth
