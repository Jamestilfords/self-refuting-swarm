"""G4 relabels a G3 failure. It is not its own look and cannot be skipped alone."""

from __future__ import annotations

from skeptic.critic.config import CriticConfig
from skeptic.critic.er1 import benefit_verdict
from skeptic.critic.types import GateResult

_NO_EFFECT = frozenset({"NEUTRAL", "HARMFUL"})
_UNDERPOWERED = frozenset({"USEFUL", "INCONCLUSIVE"})


def gate_g4(ci_low: float, ci_high: float, se_combined: float, config: CriticConfig) -> GateResult:
    label = benefit_verdict(ci_low, ci_high, band=config.g4_band)
    if label not in _NO_EFFECT | _UNDERPOWERED:
        raise ValueError(f"unexpected benefit_verdict: {label}")
    mdd = None if label in _NO_EFFECT else float(config.mdd_z * se_combined)
    return GateResult(
        "G4",
        label,
        {"benefit": label, "mdd": mdd, "se_combined": float(se_combined)},
    )


def verdict_for(label: str) -> str:
    if label in _NO_EFFECT:
        return "NO EFFECT"
    if label in _UNDERPOWERED:
        return "UNDERPOWERED"
    raise ValueError(label)
