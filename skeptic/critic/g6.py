"""G6 ceiling. A two-parent proposal may not beat the in-sample linear fusion of its parents.

linear_fusion always scores a "val" split and a "test" split. The critic has no
test arrays. Both keys are the same validation arrays, and only fusion_val_auroc
is read. Every *test* field is ignored.
"""

from __future__ import annotations

from skeptic.critic.config import CriticConfig
from skeptic.critic.er1 import linear_fusion
from skeptic.critic.types import BaselineSet, GateResult, ScoreSet


def gate_g6(candidate: ScoreSet, baselines: BaselineSet, config: CriticConfig) -> GateResult:
    if len(candidate.parents) < 2:
        return GateResult("G6", "pass", {"applicable": False})
    if len(candidate.parents) > 2:
        raise ValueError("G6 accepts at most two parents")
    try:
        first = baselines.resolve_parent(candidate.parents[0], candidate.arm)
        second = baselines.resolve_parent(candidate.parents[1], candidate.arm)
    except KeyError as exc:
        raise ValueError(f"parent {exc} is not registered on arm {candidate.arm}") from exc

    ceiling, cand_auroc, excess = [], [], []
    for index, cand_id in enumerate(candidate.val_id):
        # Same objects under "val" and "test": there is no held-out array here.
        fused = linear_fusion(
            {"val": first.val_id[index], "test": first.val_id[index]},
            {"val": first.val_oos[index], "test": first.val_oos[index]},
            {"val": second.val_id[index], "test": second.val_id[index]},
            {"val": second.val_oos[index], "test": second.val_oos[index]},
            n_angles=config.g6_angles,
        )
        cap = float(fused["fusion_val_auroc"])
        score = float(candidate.seed_aurocs()[index])
        ceiling.append(cap)
        cand_auroc.append(score)
        excess.append(score - cap)
    fail = any(value > config.g6_tol for value in excess)
    return GateResult(
        "G6",
        "fail" if fail else "pass",
        {"applicable": True, "ceiling": ceiling, "cand_auroc": cand_auroc, "excess": excess},
    )
