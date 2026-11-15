"""G2 matched baseline. Fail (SCORE-MATCHED) and choose the comparator for G3."""

from __future__ import annotations

import math

import numpy as np

from skeptic.critic.config import CriticConfig
from skeptic.critic.er1 import spearman
from skeptic.critic.seeds import majority
from skeptic.critic.types import BaselineSet, GateResult, ScoreSet


def _finite_rho(value: float) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _mean_rho(values: list[float | None]) -> float:
    finite = [value for value in values if value is not None]
    if not finite:
        return -math.inf
    return sum(finite) / len(finite)


def _matched_leader(names: list[str], rhos: dict[str, list[float | None]], thresh: float) -> str | None:
    """Baseline matched on a majority of seeds. None when nobody is.

    Ties break by most matched seeds, then highest mean rho, then name.
    """
    if not names:
        return None
    flags = {
        name: [rho is not None and rho >= thresh for rho in rhos[name]]
        for name in names
    }
    leaders = [name for name in names if majority(flags[name])]
    if not leaders:
        return None
    counts = {name: sum(flags[name]) for name in leaders}
    best_count = max(counts.values())
    leaders = [name for name in leaders if counts[name] == best_count]
    best_mean = max(_mean_rho(rhos[name]) for name in leaders)
    leaders = [name for name in leaders if _mean_rho(rhos[name]) == best_mean]
    return min(leaders)


def _comparator_key(item: tuple[str, ScoreSet]):
    kind, score = item
    # Higher AUROC wins. A tie goes to the incumbent, then to name order.
    return (-score.mean_auroc(), 0 if kind == "incumbent" else 1, score.name)


_ARMS = ("ce", "edl")


def score_family(name: str) -> str | None:
    """Stem of a registered score, so ``d_port_edl`` and ``d_port_ce`` share ``d_port``."""
    for arm in _ARMS:
        suffix = f"_{arm}"
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def family_members(name: str) -> frozenset[str]:
    family = score_family(name)
    if family is None:
        return frozenset()
    return frozenset(f"{family}_{arm}" for arm in _ARMS)


def gate_g2(
    candidate: ScoreSet,
    incumbent: ScoreSet,
    baselines: BaselineSet,
    config: CriticConfig,
    exclude_baselines: frozenset[str] = frozenset(),
    g2_mode: str = "stream",
) -> GateResult:
    if config.seed_majority != "ceil_half":
        raise ValueError(f"unknown seed_majority rule: {config.seed_majority}")
    if g2_mode not in {"stream", "scorecard"}:
        raise ValueError(f"g2_mode must be 'stream' or 'scorecard', got {g2_mode!r}")
    family_excluded = family_members(candidate.name) if g2_mode == "scorecard" else frozenset()
    names = [
        name
        for name in baselines.g2_names
        if name not in exclude_baselines and name != candidate.name and name not in family_excluded
    ]
    max_rho: dict[str, list[float | None]] = {}
    for name in names:
        baseline = baselines.scores[name]
        rhos: list[float | None] = []
        for cand_id, cand_oos, base_id, base_oos in zip(
            candidate.val_id, candidate.val_oos, baseline.val_id, baseline.val_oos
        ):
            rho = spearman(np.concatenate([cand_id, cand_oos]), np.concatenate([base_id, base_oos]))
            rhos.append(_finite_rho(rho))
        max_rho[name] = rhos

    matched_by = _matched_leader(names, max_rho, config.g2_rho_thresh)
    gain_inc = float(candidate.mean_auroc() - incumbent.mean_auroc())
    if names:
        best_baseline = min(names, key=lambda name: (-baselines.scores[name].mean_auroc(), name))
        gain_base = float(candidate.mean_auroc() - baselines.scores[best_baseline].mean_auroc())
    else:
        best_baseline = None
        gain_base = None

    weak_fail = bool(
        g2_mode == "stream"
        and config.g2_weak_incumbent_check
        and gain_base is not None
        and gain_inc > 0.0
        and gain_base <= 0.0
    )
    if g2_mode == "scorecard":
        comparator = incumbent
    else:
        pool = [("incumbent", incumbent)] + [("baseline", baselines.scores[name]) for name in names]
        comparator = min(pool, key=_comparator_key)[1]
    return GateResult(
        "G2",
        "fail" if matched_by is not None or weak_fail else "pass",
        {
            "mode": g2_mode,
            "family_excluded": sorted(family_excluded),
            "max_rho": max_rho,
            "matched_by": matched_by,
            "gain_inc": gain_inc,
            "gain_base": gain_base,
            "best_baseline": best_baseline,
            "comparator": comparator.name,
        },
    )
