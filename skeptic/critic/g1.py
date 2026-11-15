"""G1 materiality. Fail (INERT) when the ranking barely moves in a majority of seeds."""

from __future__ import annotations

from skeptic.critic.config import CriticConfig
from skeptic.critic.er1 import materiality, pair_decomposition
from skeptic.critic.seeds import majority
from skeptic.critic.types import GateResult, ScoreSet


def gate_g1(candidate: ScoreSet, incumbent: ScoreSet, config: CriticConfig) -> GateResult:
    if config.seed_majority != "ceil_half":
        raise ValueError(f"unknown seed_majority rule: {config.seed_majority}")
    flip_rate, fixed, broken, inert = [], [], [], []
    for ident, oos, cand_id, cand_oos in zip(
        incumbent.val_id, incumbent.val_oos, candidate.val_id, candidate.val_oos
    ):
        parts = pair_decomposition(ident, oos, cand_id, cand_oos)
        flip_rate.append(float(parts["flip_rate"]))
        fixed.append(float(parts["fixed"]))
        broken.append(float(parts["broken"]))
        inert.append(materiality(parts["flip_rate"], config.g1_flip_thresh) == "INERT")
    fail = majority(inert)
    return GateResult(
        "G1",
        "fail" if fail else "pass",
        {
            "flip_rate": flip_rate,
            "fixed": fixed,
            "broken": broken,
            "inert_seeds": int(sum(inert)),
        },
    )
