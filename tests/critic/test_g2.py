"""G2: rank match at the threshold, self-exclusion, weak incumbent, comparator ties."""

from __future__ import annotations

import numpy as np

from skeptic.critic.config import CriticConfig
from skeptic.critic.g2 import gate_g2
from skeptic.critic.types import BaselineSet, ScoreSet


def _set(name: str, ident, oos, arm: str = "ce") -> ScoreSet:
    return ScoreSet(
        name=name,
        arm=arm,
        val_id=(np.asarray(ident, dtype=np.float64),),
        val_oos=(np.asarray(oos, dtype=np.float64),),
    )


def _baselines(scores: dict[str, ScoreSet], names: tuple[str, ...]) -> BaselineSet:
    return BaselineSet(cell="synth/bert", scores=scores, g2_names=names, plain_ece={"ce": (0.0,)})


def test_rho_at_threshold_matches(monkeypatch):
    monkeypatch.setattr("skeptic.critic.g2.spearman", lambda a, b: 0.999)
    inc = _set("inc", [0, 1, 2], [3, 4, 5])
    base = _set("energy_ce", [0, 1, 2], [2, 3, 4])
    cand = _set("cand", [5, 4, 3], [2, 1, 0])
    result = gate_g2(cand, inc, _baselines({"energy_ce": base}, ("energy_ce",)), CriticConfig())
    assert result.outcome == "fail"
    assert result.detail["matched_by"] == "energy_ce"


def test_rho_below_threshold_is_not_matched(monkeypatch):
    monkeypatch.setattr("skeptic.critic.g2.spearman", lambda a, b: 0.998999)
    inc = _set("inc", [0, 0, 0], [1, 1, 1])
    base = _set("energy_ce", [0, 0, 0], [1, 1, 1])
    cand = _set("cand", [0, 0, 0], [1, 1, 1])
    result = gate_g2(cand, inc, _baselines({"energy_ce": base}, ("energy_ce",)), CriticConfig())
    assert result.detail["matched_by"] is None
    assert result.outcome == "pass"


def test_nan_rho_is_not_matched():
    inc = _set("inc", [0, 1, 2, 3], [4, 5, 6, 7])
    base = _set("energy_ce", [0, 1, 2, 3], [4, 5, 6, 7])
    cand = _set("cand", [1, 1, 1, 1], [1, 1, 1, 1])
    result = gate_g2(cand, inc, _baselines({"energy_ce": base}, ("energy_ce",)), CriticConfig())
    assert result.detail["max_rho"]["energy_ce"] == [None]
    assert result.detail["matched_by"] is None
    assert result.outcome == "pass"


def test_candidate_name_excludes_itself(monkeypatch):
    monkeypatch.setattr("skeptic.critic.g2.spearman", lambda a, b: 1.0)
    inc = _set("inc", [0, 1], [2, 3])
    msp = _set("msp_ce", [0, 1], [2, 3])
    energy = _set("energy_ce", [1, 0], [0, 1])
    cand = _set("msp_ce", [0, 1], [2, 3])
    names = ("energy_ce", "msp_ce")
    result = gate_g2(cand, inc, _baselines({"msp_ce": msp, "energy_ce": energy}, names), CriticConfig())
    assert "msp_ce" not in result.detail["max_rho"]
    assert "energy_ce" in result.detail["max_rho"]


def test_exclude_baselines_drops_rho_and_best(monkeypatch):
    monkeypatch.setattr("skeptic.critic.g2.spearman", lambda a, b: 0.0)
    inc = _set("inc", [0, 0, 0], [0, 0, 1])  # weak
    strong = _set("energy_ce", [0, 0, 0], [1, 1, 1])
    weaker = _set("msp_ce", [0, 0, 0], [0, 1, 1])
    cand = _set("cand", [0, 0, 0], [0.2, 1, 1])
    baselines = _baselines({"energy_ce": strong, "msp_ce": weaker}, ("energy_ce", "msp_ce"))
    kept = gate_g2(cand, inc, baselines, CriticConfig())
    dropped = gate_g2(cand, inc, baselines, CriticConfig(), exclude_baselines=frozenset({"energy_ce"}))
    assert kept.detail["best_baseline"] == "energy_ce"
    assert "energy_ce" not in dropped.detail["max_rho"]
    assert dropped.detail["best_baseline"] == "msp_ce"


def test_weak_incumbent_fails():
    # Candidate beats a weak incumbent and does not beat the best baseline.
    inc = _set("inc", [0, 1, 2], [0, 1, 2])
    best = _set("energy_ce", [0, 0, 0], [1, 1, 1])
    other = _set("msp_ce", [0, 0, 1], [0, 1, 1])
    cand = _set("cand", [0, 0, 0.2], [0.1, 1, 1])
    result = gate_g2(
        cand,
        inc,
        _baselines({"energy_ce": best, "msp_ce": other}, ("energy_ce", "msp_ce")),
        CriticConfig(),
    )
    assert result.detail["gain_inc"] > 0
    assert result.detail["gain_base"] <= 0
    assert result.detail["matched_by"] is None
    assert result.outcome == "fail"


def test_comparator_tie_prefers_incumbent_then_name():
    perfect_id, perfect_oos = [0, 0, 0], [1, 1, 1]
    inc = _set("inc", perfect_id, perfect_oos)
    tied = _set("msp_ce", perfect_id, perfect_oos)
    worse = _set("energy_ce", [0, 1, 2], [0, 1, 2])
    cand = _set("cand", [0, 0, 1], [0, 1, 2])
    result = gate_g2(
        cand,
        inc,
        _baselines({"msp_ce": tied, "energy_ce": worse}, ("msp_ce", "energy_ce")),
        CriticConfig(),
    )
    assert result.detail["comparator"] == "inc"

    weak_inc = _set("inc", [0, 1, 2], [0, 1, 2])
    full = _set("d_full_ce", perfect_id, perfect_oos)
    port = _set("d_port_ce", [0, 0, 0], [1, 1, 1])
    result = gate_g2(
        cand,
        weak_inc,
        _baselines({"d_full_ce": full, "d_port_ce": port}, ("d_port_ce", "d_full_ce")),
        CriticConfig(),
    )
    assert result.detail["comparator"] == "d_full_ce"


def test_scorecard_mode_drops_both_arms_and_keeps_the_incumbent():
    inc = _set("energy_ce", [0, 1, 2], [0, 1, 2])
    twin = _set("d_port_ce", [0, 0, 0], [1, 1, 1])
    same = _set("d_port_edl", [0, 0, 0], [1, 1, 1], arm="edl")
    other = _set("msp_ce", [3, 2, 1], [0, 0, 0])
    cand = _set("d_port_edl", [0, 0, 0], [1, 1, 1], arm="edl")
    names = ("d_port_ce", "d_port_edl", "msp_ce")
    baselines = _baselines(
        {"energy_ce": inc, "d_port_ce": twin, "d_port_edl": same, "msp_ce": other},
        names,
    )
    stream = gate_g2(cand, inc, baselines, CriticConfig(), g2_mode="stream")
    scorecard = gate_g2(cand, inc, baselines, CriticConfig(), g2_mode="scorecard")
    assert stream.outcome == "fail"
    assert stream.detail["matched_by"] == "d_port_ce"
    assert set(scorecard.detail["family_excluded"]) == {"d_port_ce", "d_port_edl"}
    assert "d_port_ce" not in scorecard.detail["max_rho"]
    assert scorecard.detail["matched_by"] is None
    assert scorecard.detail["comparator"] == "energy_ce"
    assert scorecard.outcome == "pass"


def test_scorecard_mode_does_not_apply_the_weak_incumbent_check():
    inc = _set("inc", [0, 1, 2], [0, 1, 2])
    best = _set("energy_ce", [0, 0, 0], [1, 1, 1])
    cand = _set("cand", [0, 0, 0.2], [0.1, 1, 1])
    baselines = _baselines({"energy_ce": best}, ("energy_ce",))
    result = gate_g2(cand, inc, baselines, CriticConfig(), g2_mode="scorecard")
    assert result.detail["gain_inc"] > 0
    assert result.detail["gain_base"] <= 0
    assert result.outcome == "pass"
    assert result.detail["comparator"] == "inc"
