"""Gate pipeline. The first failure is the verdict. LORD++ advances at most once."""

from __future__ import annotations

from skeptic.critic.config import DEFAULT_CONFIG, CriticConfig
from skeptic.critic.g1 import gate_g1
from skeptic.critic.g2 import gate_g2
from skeptic.critic.g3 import benefit_stats
from skeptic.critic.g4 import gate_g4, verdict_for
from skeptic.critic.g5 import gate_g5
from skeptic.critic.g6 import gate_g6
from skeptic.critic.lord import LordPlusPlus
from skeptic.critic.types import BaselineSet, Decision, ScoreSet, Verdict
from skeptic.critic.cache import StatsCache

_SKIPPABLE = frozenset({"G1", "G2", "G3", "G5", "G6"})


def verdict(
    candidate: ScoreSet,
    incumbent: ScoreSet,
    baselines: BaselineSet,
    fdr_state: LordPlusPlus,
    cell: str,
    *,
    config: CriticConfig = DEFAULT_CONFIG,
    skip: frozenset[str] = frozenset(),
    exclude_baselines: frozenset[str] = frozenset(),
    cache: StatsCache | None = None,
) -> Decision:
    if cell != baselines.cell:
        raise ValueError(f"cell {cell!r} does not match baselines.cell {baselines.cell!r}")
    unknown = set(skip) - _SKIPPABLE
    if unknown:
        raise ValueError(f"cannot skip {sorted(unknown)}")
    if candidate.status == "guard":
        return _decision(
            candidate, cell, incumbent.name, None, None, None, None, None, None,
            {"lock": "fail"}, Verdict.BUG, None, {"lock": {"status": "guard"}}, config,
        )
    if candidate.status in {"malformed", "timeout"}:
        raise ValueError(f"candidate status {candidate.status!r} is not a verdict")
    if candidate.status != "ok":
        raise ValueError(f"candidate status {candidate.status!r} is not a verdict")
    if incumbent.status != "ok":
        raise ValueError("incumbent status is not ok")
    _same_shape(candidate, incumbent)

    gates: dict[str, str] = {}
    detail: dict = {}

    if "G1" in skip:
        gates["G1"] = "skipped"
    else:
        g1 = gate_g1(candidate, incumbent, config)
        gates["G1"] = g1.outcome
        detail["G1"] = g1.detail
        if g1.outcome == "fail":
            return _decision(
                candidate, cell, incumbent.name, None, None, None, None, None, None,
                gates, Verdict.INERT, None, detail, config,
            )

    if "G2" in skip:
        gates["G2"] = "skipped"
        comparator = incumbent
    else:
        g2 = gate_g2(candidate, incumbent, baselines, config, exclude_baselines)
        gates["G2"] = g2.outcome
        detail["G2"] = g2.detail
        comparator = _lookup_comparator(g2.detail["comparator"], incumbent, baselines)
        if g2.outcome == "fail":
            return _decision(
                candidate, cell, incumbent.name, comparator.name, None, None, None, None, None,
                gates, Verdict.SCORE_MATCHED, None, detail, config,
            )

    _same_shape(candidate, comparator)
    if "G3" in skip:
        gates["G3"] = "skipped"
        delta = float((candidate.seed_aurocs() - comparator.seed_aurocs()).mean())
        detail["G3"] = {"seed_deltas": [float(v) for v in candidate.seed_aurocs() - comparator.seed_aurocs()]}
        if delta <= 0.0:
            return _decision(
                candidate, cell, incumbent.name, comparator.name, delta, None, None, None, None,
                gates, Verdict.NO_EFFECT, None, detail, config,
            )
    else:
        stats = benefit_stats(candidate, comparator, config, cell, cache)
        delta = stats["delta"]
        if delta <= 0.0:
            alpha_t = fdr_state.level()
            fdr_state.test(1.0)
            gates["G3"] = "fail"
            detail["G3"] = _g3_detail(stats)
            return _decision(
                candidate, cell, incumbent.name, comparator.name, delta, None, 1.0, alpha_t, fdr_state.t,
                gates, Verdict.NO_EFFECT, None, detail, config,
            )
        p = stats["p_one_sided"]
        alpha_t = fdr_state.level()
        rejected = fdr_state.test(p)
        detail["G3"] = _g3_detail(stats)
        ci = (stats["ci95"][0], stats["ci95"][1])
        if not rejected:
            gates["G3"] = "fail"
            g4 = gate_g4(ci[0], ci[1], stats["se_combined"], config)
            gates["G4"] = g4.outcome
            detail["G4"] = g4.detail
            label = verdict_for(g4.outcome)
            mdd = g4.detail["mdd"] if label == "UNDERPOWERED" else None
            return _decision(
                candidate, cell, incumbent.name, comparator.name, delta, ci, p, alpha_t, fdr_state.t,
                gates, Verdict(label), mdd, detail, config,
            )
        gates["G3"] = "pass"

    reached_g3 = "G3" not in skip
    # Values carried forward when G3 passed, or when G3 was skipped and delta > 0.
    if reached_g3:
        trailing = {
            "delta": delta,
            "ci": ci,
            "p": p,
            "alpha_t": alpha_t,
            "fdr_t": fdr_state.t,
        }
    else:
        trailing = {"delta": delta, "ci": None, "p": None, "alpha_t": None, "fdr_t": None}

    if "G5" in skip:
        gates["G5"] = "skipped"
    else:
        g5 = gate_g5(candidate, baselines, config)
        gates["G5"] = g5.outcome
        detail["G5"] = g5.detail
        if g5.outcome == "fail":
            return _decision(
                candidate, cell, incumbent.name, comparator.name,
                trailing["delta"], trailing["ci"], trailing["p"], trailing["alpha_t"], trailing["fdr_t"],
                gates, Verdict.MISCALIBRATES, None, detail, config,
            )

    if "G6" in skip:
        gates["G6"] = "skipped"
    else:
        g6 = gate_g6(candidate, baselines, config)
        gates["G6"] = g6.outcome
        detail["G6"] = g6.detail
        if g6.outcome == "fail":
            return _decision(
                candidate, cell, incumbent.name, comparator.name,
                trailing["delta"], trailing["ci"], trailing["p"], trailing["alpha_t"], trailing["fdr_t"],
                gates, Verdict.BUG, None, detail, config,
            )

    return _decision(
        candidate, cell, incumbent.name, comparator.name,
        trailing["delta"], trailing["ci"], trailing["p"], trailing["alpha_t"], trailing["fdr_t"],
        gates, Verdict.ACCEPT, None, detail, config,
    )


def _g3_detail(stats: dict) -> dict:
    return {
        "seed_deltas": stats["seed_deltas"],
        "se_boot": stats["se_boot"],
        "se_combined": stats["se_combined"],
        "ci95": stats["ci95"],
        "ci95_boot": stats["ci95_boot"],
        "p_two_sided_boot": stats["p_two_sided_boot"],
    }


def _lookup_comparator(name: str, incumbent: ScoreSet, baselines: BaselineSet) -> ScoreSet:
    if name == incumbent.name:
        return incumbent
    return baselines.scores[name]


def _same_shape(candidate: ScoreSet, other: ScoreSet) -> None:
    if len(candidate.val_id) != len(other.val_id):
        raise ValueError("candidate and comparator have different seed counts")
    for cand_id, cand_oos, other_id, other_oos in zip(
        candidate.val_id, candidate.val_oos, other.val_id, other.val_oos
    ):
        if cand_id.shape != other_id.shape or cand_oos.shape != other_oos.shape:
            raise ValueError("candidate and comparator score lengths differ")


def _decision(
    candidate, cell, incumbent, comparator, delta, ci, p, alpha_t, fdr_t,
    gates, verdict_label, mdd, detail, config,
) -> Decision:
    full = dict(detail)
    full["config_digest"] = config.digest()
    return Decision(
        sha=candidate.name,
        cell=cell,
        incumbent=incumbent,
        comparator=comparator,
        delta_val=None if delta is None else float(delta),
        ci=None if ci is None else (float(ci[0]), float(ci[1])),
        p=None if p is None else float(p),
        alpha_t=None if alpha_t is None else float(alpha_t),
        fdr_t=None if fdr_t is None else int(fdr_t),
        gates=dict(gates),
        verdict=verdict_label,
        mdd=None if mdd is None else float(mdd),
        detail=full,
    )
