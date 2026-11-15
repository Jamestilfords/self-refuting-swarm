"""E5 R0: the keep-if-better validation ratchet."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _mean(row: dict) -> float:
    values = row.get("val", {}).get("auroc") if isinstance(row.get("val"), dict) else None
    if not isinstance(values, list) or len(values) != 5 \
            or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError("R0 requires five finite validation AUROCs")
    return sum(values) / 5


@dataclass
class Ratchet:
    incumbent_sha: str
    incumbent_mean: float

    @classmethod
    def from_baseline(cls, row: dict) -> "Ratchet":
        if row.get("status") != "ok" or row.get("stream") != "baseline":
            raise ValueError("R0 starting incumbent must be a scored baseline")
        return cls(row["sha"], _mean(row))

    def step(self, row: dict) -> dict:
        before = self.incumbent_sha
        decision = {"rule": "R0", "sha": row["sha"], "incumbent": before,
                    "comparator": before, "delta_val": None,
                    "verdict": "REJECT", "reason": row.get("status", "missing_status")}
        if row.get("status") != "ok":
            return decision
        candidate = _mean(row)
        delta = candidate - self.incumbent_mean
        decision["delta_val"] = round(delta, 9)
        decision["reason"] = "nonpositive_gain"
        if delta > 0:
            decision["verdict"] = "ACCEPT"
            decision["reason"] = "positive_gain"
            self.incumbent_sha, self.incumbent_mean = row["sha"], candidate
        return decision
