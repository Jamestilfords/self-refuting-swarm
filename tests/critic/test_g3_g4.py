"""G3 short-circuit and the four G4 labels, with a stubbed bootstrap."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from skeptic.critic.config import CriticConfig
from skeptic.critic.g4 import gate_g4
from skeptic.critic.lord import LordPlusPlus
from skeptic.critic.types import BaselineSet, ScoreSet
from skeptic.critic.verdict import verdict

FAST = CriticConfig(bootstrap_B=200)


def _set(name, ident, oos) -> ScoreSet:
    return ScoreSet(
        name=name,
        arm="ce",
        val_id=(np.asarray(ident, dtype=np.float64),) * 5,
        val_oos=(np.asarray(oos, dtype=np.float64),) * 5,
    )


def _cell(inc: ScoreSet, cand: ScoreSet):
    energy = _set("energy_ce", [0, 0, 0, 0], [0, 0, 1, 1])
    others = {
        "msp_ce": _set("msp_ce", [0, 0, 0, 0], [0, 1, 0, 1]),
        "d_port_ce": _set("d_port_ce", [0, 1, 0, 1], [0, 1, 0, 1]),
        "d_full_ce": _set("d_full_ce", [1, 0, 1, 0], [1, 0, 1, 0]),
        "energy_edl": _set("energy_edl", [0, 0, 1, 1], [0, 0, 1, 1]),
        "msp_edl": _set("msp_edl", [0, 1, 2, 3], [0, 1, 2, 3]),
        "d_port_edl": _set("d_port_edl", [3, 2, 1, 0], [3, 2, 1, 0]),
        "d_full_edl": _set("d_full_edl", [0, 0, 0, 1], [0, 0, 0, 1]),
    }
    scores = {"energy_ce": energy, **others}
    # The hand-built incumbent is the comparator. Keep it inside the set when
    # its name collides with a baseline; otherwise G2 still sees the baselines.
    if inc.name in scores:
        scores[inc.name] = inc
    baselines = BaselineSet(
        cell="hand/bert",
        scores=scores,
        g2_names=tuple(scores),
        plain_ece={"ce": (0.0,) * 5, "edl": (0.0,) * 5},
    )
    return baselines


def test_nonpositive_delta_skips_bootstrap_and_spends_one_step(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("paired_bootstrap should not run when delta <= 0")

    monkeypatch.setattr("skeptic.critic.er1.paired_bootstrap", boom)
    inc = _set("energy_ce", [0, 0, 0, 0], [1, 1, 1, 1])
    cand = _set("cand", [0, 0, 0, 0], [0, 0, 0, 0])
    fdr = LordPlusPlus()
    decision = verdict(cand, inc, _cell(inc, cand), fdr, "hand/bert", config=FAST)
    assert decision.verdict.value == "NO EFFECT"
    assert decision.p == 1.0
    assert decision.gates["G3"] == "fail"
    assert "G4" not in decision.gates
    assert fdr.t == 1
    assert fdr.rejections == ()


def test_benefit_labels_map_to_verdicts(monkeypatch):
    calls = {"boot": 0}

    def fake_boot(id_a, oos_a, id_b, oos_b, **kwargs):
        calls["boot"] += 1
        from skeptic.critic.er1 import auroc_scores

        delta = float(np.mean([
            auroc_scores(id_a[i], oos_a[i]) - auroc_scores(id_b[i], oos_b[i]) for i in range(len(id_a))
        ]))
        return {"delta": delta, "ci_low": -1.0, "ci_high": 1.0, "se": 1.0, "p": 0.5}

    bounds = {"lo": 0.0, "hi": 0.0, "se": 1.0}

    def fake_interval(delta, se_boot, seed_deltas):
        return {
            "delta": float(delta),
            "se_boot": float(se_boot),
            "se_combined": bounds["se"],
            "ci95_low": bounds["lo"],
            "ci95_high": bounds["hi"],
        }

    monkeypatch.setattr("skeptic.critic.er1.paired_bootstrap", fake_boot)
    monkeypatch.setattr("skeptic.critic.er1.combined_interval", fake_interval)

    inc = _set("energy_ce", [0, 1, 2, 3], [0.5, 1.5, 2.5, 3.5])
    cand = _set("cand", [0, 0, 0, 0], [1, 1, 1, 1])
    baselines = _cell(inc, cand)
    cases = {
        "NEUTRAL": ((-0.0004, 0.0004), "NO EFFECT"),
        "HARMFUL": ((-0.05, -0.002), "NO EFFECT"),
        "USEFUL": ((0.002, 0.05), "UNDERPOWERED"),
        "INCONCLUSIVE": ((-0.01, 0.02), "UNDERPOWERED"),
    }
    for label, ((lo, hi), expected) in cases.items():
        bounds.update(lo=lo, hi=hi, se=0.2)
        decision = verdict(cand, inc, baselines, LordPlusPlus(), "hand/bert", config=FAST)
        assert decision.gates["G4"] == label
        assert decision.verdict.value == expected
        if expected == "UNDERPOWERED":
            z = float(norm.ppf(0.95) + norm.ppf(0.80))
            assert decision.mdd == z * bounds["se"]
    assert calls["boot"] == 4


def test_gate_g4_mdd_uses_scipy_z():
    result = gate_g4(0.002, 0.05, 0.2, FAST)
    z = float(norm.ppf(0.95) + norm.ppf(0.80))
    assert result.outcome == "USEFUL"
    assert result.detail["mdd"] == z * 0.2
