"""E5 R2 adapter for Ryu's validation-only Decision and LORD++ state.

The caller supplies one fresh LordPlusPlus object per cell and ordering. Ryu's
verdict function mutates that object only when its G3 test is reached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from skeptic.rules.common import accept_gates_ok

VerdictFn = Callable[[dict, str, Any, int], Any]
_VERDICTS = frozenset({"ACCEPT", "INERT", "SCORE-MATCHED", "NO EFFECT",
                       "UNDERPOWERED", "MISCALIBRATES", "BUG"})


@dataclass
class Critic:
    incumbent_sha: str
    state: Any
    verdict_fn: VerdictFn
    rule: str = "R2"

    def step(self, row: dict, position: int) -> dict:
        before = self.incumbent_sha
        if row.get("status") not in ("ok", "io"):
            return {"rule": self.rule, "sha": row["sha"], "incumbent": before,
                    "comparator": None, "delta_val": None, "ci": None, "p": None,
                    "alpha_t": None, "mdd": None, "gates": {},
                    "verdict": "REJECT", "reason": row.get("status", "missing_status")}

        result = self.verdict_fn(row, before, self.state, position)
        if not callable(getattr(result, "to_row", None)):
            raise ValueError("R2 callback must return Ryu's Decision object")
        item = result.to_row(idx=position)
        if not isinstance(item, dict) or item.get("sha") != row["sha"] \
                or item.get("incumbent") != before or item.get("cell") != row.get("cell") \
                or item.get("idx") != position or item.get("verdict") not in _VERDICTS \
                or not isinstance(item.get("gates"), dict):
            raise ValueError("Ryu Decision violated the R2 contract")
        if item["verdict"] == "ACCEPT" and not accept_gates_ok(self.rule, item["gates"]):
            raise ValueError("R2 cannot accept unless every applicable critic gate passes")
        if item["verdict"] == "ACCEPT":
            self.incumbent_sha = row["sha"]
        return {"rule": self.rule, **item}
