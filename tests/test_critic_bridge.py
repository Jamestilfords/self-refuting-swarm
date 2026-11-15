"""The replay adapter reuses the merged critic's inputs and state semantics."""
from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace
from skeptic.rules.critic_bridge import ensure_er1_path, critic_factory
ensure_er1_path()
from skeptic.critic.config import CriticConfig
from skeptic.critic.types import ScoreSet
from skeptic.critic.lord import LordPlusPlus
from skeptic.critic.verdict import verdict
from tests.fixtures.critic.synth import make_cell, strong


def test_adapter_uses_sha_incumbent_and_fresh_lord_state():
    synth = make_cell()
    base, good, guard = "0" * 12, "1" * 12, "2" * 12
    inputs = SimpleNamespace(cell=synth.cell, baselines=synth.baselines,
        candidates={base: replace(synth.incumbent, name=base), good: replace(strong(synth), name=good),
                    guard: ScoreSet(name=guard, arm="ce", val_id=(), val_oos=(), status="guard")})
    make = critic_factory({synth.cell: inputs}, CriticConfig(bootstrap_B=200))
    first = make(synth.cell, 0, base)
    decision = first.step({"sha": good, "cell": synth.cell, "status": "ok"}, 0)
    assert decision["verdict"] == "ACCEPT" and decision["incumbent"] == base
    assert first.incumbent_sha == good and first.state.t == 1
    blocked = first.step({"sha": guard, "cell": synth.cell, "status": "io"}, 1)
    assert blocked["verdict"] == "BUG" and blocked["gates"] == {"lock": "fail"}
    assert first.state.t == 1
    second = make(synth.cell, 1, base)
    assert second.state.t == 0 and second.incumbent_sha == base


def test_cached_gates_preserve_decisions_and_fdr_state(monkeypatch):
    synth = make_cell()
    candidate = strong(synth)
    cache = {}
    first = verdict(candidate, synth.incumbent, synth.baselines, LordPlusPlus(), synth.cell,
                    config=CriticConfig(bootstrap_B=40), gate_cache=cache)
    assert {key[0] for key in cache} == {"G1", "G2"}

    def unexpected(*_args, **_kwargs):
        raise AssertionError("repeated gate was recomputed")

    verdict_module = import_module("skeptic.critic.verdict")
    monkeypatch.setattr(verdict_module, "gate_g1", unexpected)
    monkeypatch.setattr(verdict_module, "gate_g2", unexpected)
    second = verdict(candidate, synth.incumbent, synth.baselines, LordPlusPlus(), synth.cell,
                     config=CriticConfig(bootstrap_B=40), gate_cache=cache)
    assert first.to_json() == second.to_json()
