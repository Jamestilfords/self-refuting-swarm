"""The eight registered rediscovery pairs and their strict and lenient grades.

Each pair is scored by the frozen critic. Scorecard mode excludes the
candidate's score family in both arms, turns off the weak-incumbent check, and
compares the candidate with the incumbent. Stream mode is reported beside it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from skeptic.critic.config import DEFAULT_CONFIG, CriticConfig
from skeptic.critic.lord import LordPlusPlus
from skeptic.critic.types import BaselineSet, ScoreSet
from skeptic.critic.verdict import verdict
from skeptic.rules.critic_bridge import drop_ignored_parents

PairKey = tuple[str, str]


@dataclass(frozen=True)
class ScorecardPair:
    number: int
    cell: str
    candidate: str
    incumbent: str
    strict: frozenset[str]
    lenient: frozenset[str]
    note: str


def _pair(number, cell, candidate, incumbent, strict, lenient, note) -> ScorecardPair:
    return ScorecardPair(
        number,
        cell,
        candidate,
        incumbent,
        frozenset(strict),
        frozenset(lenient),
        note,
    )


PAIRS: tuple[ScorecardPair, ...] = (
    _pair(1, "clinc150/bert", "vacuity_edl", "msp_ce", ("SCORE-MATCHED",), ("SCORE-MATCHED",),
          "Vacuity minus CE MSP is a scoring-rule effect, and vacuity matches energy."),
    _pair(2, "clinc150/bert", "d_port_edl", "d_port_edl_unwhitened", ("INERT",), ("INERT",),
          "Whitening is a no-op for tied Mahalanobis. The candidate is the registered score."),
    _pair(3, "clinc150/bert", "qsigmoid_mul_d_port_edl", "vacuity_edl",
          ("NO EFFECT",), ("NO EFFECT", "MISCALIBRATES"),
          "Coupled vacuity is harmful, and it can also fail calibration."),
    _pair(4, "clinc150/bert", "qsigmoid_mul_d_port_edl", "d_port_edl",
          ("INERT",), ("INERT", "SCORE-MATCHED"),
          "The coupling stays within 0.001 AUROC of the density."),
    _pair(5, "clinc150/bert", "d_bottleneck_edl", "d_port_edl",
          ("ACCEPT",), ("ACCEPT", "UNDERPOWERED"),
          "No-PCA bottleneck density versus PCA-128. Likely underpowered on CLINC150."),
    _pair(6, "banking77_oos/bert", "d_port_edl", "vacuity_edl",
          ("ACCEPT",), ("ACCEPT", "UNDERPOWERED"),
          "Bottleneck density beats vacuity on Banking77-OOS."),
    _pair(7, "banking77_oos/bert", "ecdf_add_log_d_full_edl", "vacuity_edl",
          ("ACCEPT",), ("ACCEPT", "UNDERPOWERED"),
          "add_log at lambda 5 over vacuity. Likely underpowered."),
    _pair(8, "clinc150/bert", "relative_mahalanobis_edl", "d_full_edl",
          ("NO EFFECT",), ("NO EFFECT",),
          "Relative Mahalanobis loses to the full density."),
)


def grade(label: str, pair: ScorecardPair) -> str:
    """``strict`` if the verdict is the registered one, else ``lenient`` or ``miss``."""
    if label in pair.strict:
        return "strict"
    if label in pair.lenient:
        return "lenient"
    return "miss"


def match_counts(grades: Mapping[int, str]) -> dict[str, int]:
    """Strict matches, and lenient matches, each out of the eight pairs.

    A strict grade also counts as lenient. Missing pair numbers count as misses.
    """
    by_number = {pair.number: pair for pair in PAIRS}
    if set(grades) - set(by_number):
        raise ValueError(f"unknown scorecard pair {sorted(set(grades) - set(by_number))}")
    strict = 0
    lenient = 0
    for number in by_number:
        label = grades.get(number, "miss")
        if label == "strict":
            strict += 1
            lenient += 1
        elif label == "lenient":
            lenient += 1
        elif label != "miss":
            raise ValueError(f"pair {number} has grade {label!r}")
    return {"n_pairs": len(PAIRS), "strict_matches": strict, "lenient_matches": lenient}


def run_scorecard(
    scores: Mapping[PairKey, ScoreSet],
    baselines: Mapping[str, BaselineSet],
    *,
    config: CriticConfig = DEFAULT_CONFIG,
    resolved: Mapping[int, tuple[ScoreSet, ScoreSet]] | None = None,
) -> list[dict]:
    """Run each registered pair in scorecard mode and again in stream mode.

    Each call gets a fresh LORD++ state. ``scores`` is keyed by (cell, name).
    ``resolved`` supplies the candidate and incumbent for a pair number when the
    published file is addressed by proposal hash rather than by that name.
    """
    rows = []
    for pair in PAIRS:
        if resolved is not None and pair.number in resolved:
            candidate, incumbent = resolved[pair.number]
        else:
            candidate = scores[(pair.cell, pair.candidate)]
            incumbent = scores[(pair.cell, pair.incumbent)]
        candidate = drop_ignored_parents(candidate)
        cell_baselines = baselines[pair.cell]
        scorecard = verdict(
            candidate, incumbent, cell_baselines, LordPlusPlus(alpha=config.fdr_alpha, w0=config.fdr_w0),
            pair.cell, config=config, g2_mode="scorecard",
        )
        stream = verdict(
            candidate, incumbent, cell_baselines, LordPlusPlus(alpha=config.fdr_alpha, w0=config.fdr_w0),
            pair.cell, config=config, g2_mode="stream",
        )
        rows.append({
            "number": pair.number,
            "cell": pair.cell,
            "candidate": pair.candidate,
            "incumbent": pair.incumbent,
            "scorecard_verdict": scorecard.verdict.value,
            "stream_verdict": stream.verdict.value,
            "grade": grade(scorecard.verdict.value, pair),
            "delta_val": scorecard.delta_val,
            "ci_low": None if scorecard.ci is None else scorecard.ci[0],
            "ci_high": None if scorecard.ci is None else scorecard.ci[1],
            "p": scorecard.p,
            "alpha_t": scorecard.alpha_t,
            "note": pair.note,
        })
    return rows
