"""G5 calibration on in-scope validation items. Fail when ECE rises past the tolerance."""

from __future__ import annotations

from skeptic.critic.config import CriticConfig
from skeptic.critic.types import BaselineSet, GateResult, ScoreSet


def gate_g5(candidate: ScoreSet, baselines: BaselineSet, config: CriticConfig) -> GateResult:
    if candidate.ece is None:
        return GateResult("G5", "pass", {"applicable": False})
    plain = baselines.plain_ece[candidate.arm]
    delta = float(sum(candidate.ece) / len(candidate.ece) - sum(plain) / len(plain))
    return GateResult(
        "G5",
        "fail" if delta > config.g5_ece_tol else "pass",
        {"applicable": True, "delta_ece": delta, "bins": config.g5_bins},
    )
