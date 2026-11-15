"""E5 replay tests using synthetic scores; no cloud calls."""
import json
from pathlib import Path
import numpy as np
import pytest
from skeptic import store
from skeptic.rules.common import CELLS, ScoreArtifact, load_ledger, matched_orderings
from skeptic.rules.critic import Critic
from skeptic.rules.engine import combine_cells, replay_cell, write_decision_logs
from skeptic.rules.ratchet import Ratchet
from skeptic.rules.thresholdout import Thresholdout, stratified_split
BASE, GOOD, BAD = "000000000000", "111111111111", "222222222222"

class FakeDecision:
    def __init__(self, row):
        self.row = row

    def to_row(self, idx=None):
        return {**self.row, "idx": idx}

def _critic_row(sha, cell, incumbent, *, gate_g3="pass"):
    return {"sha": sha, "cell": cell, "incumbent": incumbent,
            "comparator": incumbent, "delta_val": .1, "ci": [0., .2],
            "p": .01, "alpha_t": .05, "mdd": None, "fdr_t": 1,
            "gates": {gate: (gate_g3 if gate == "G3" else "pass")
                      for gate in ("G1", "G2", "G3", "G5", "G6")},
            "verdict": "ACCEPT", "detail": {}, "er1_commit": "e877e9159d"}

def _row(sha: str, stream: str, score: float, cell: str = "clinc150/bert") -> dict:
    return {"sha": sha, "stream": stream, "status": "ok", "cell": cell,
            "val": {"auroc": [score] * 5}}

def _artifact(kind: str) -> ScoreArtifact:
    scores = np.zeros(8)
    if kind == "good":
        scores[4:] = 1
    elif kind == "bad":
        scores[:4] = 1
    return ScoreArtifact(np.tile(scores, (5, 1)), 4)

def _logs() -> dict:
    logs = {}
    for rule in ("R0", "R1", "R2"):
        logs[(rule, 0)] = [
            {"schema_version": 1, "rule": rule, "cell": cell, "ordering": 0,
             "idx": 0, "sha": GOOD, "incumbent": BASE, "verdict": "ACCEPT",
             **({"gates": {gate: "pass" for gate in ("G1", "G2", "G3", "G5", "G6")}}
                if rule == "R2" else {})}
            for cell in CELLS
        ]
    return logs

def _freeze(tmp_path: Path) -> str:
    (tmp_path / "PREREGISTRATION.md").write_text("Status: FROZEN\n", encoding="utf-8")
    write_decision_logs(tmp_path, _logs(), ordering_count=1)
    return store.write_decision_hash(tmp_path)

def test_decision_hash_rejects_draft_tampering_and_incomplete_rule_set(tmp_path):
    write_decision_logs(tmp_path, _logs(), ordering_count=1)
    (tmp_path / "PREREGISTRATION.md").write_text("Status: DRAFT TEMPLATE\nTODO: fix\n")
    with pytest.raises(store.GuardError, match="draft"):
        store.write_decision_hash(tmp_path)
    (tmp_path / "PREREGISTRATION.md").write_text("Status: FROZEN\n")
    digest = store.write_decision_hash(tmp_path)
    assert digest == store.verify_decision_hash(tmp_path)
    with pytest.raises(store.GuardError, match="already exists"):
        store.write_decision_hash(tmp_path)
    target = tmp_path / "runs/decisions/R1/ord_000.jsonl"
    target.write_text(target.read_text().replace(GOOD, BAD, 1))
    with pytest.raises(store.GuardError):
        store.verify_decision_hash(tmp_path)

def test_r0_r1_r2_replay_decisions_and_gate_enforcement():
    baseline, good, bad = _row(BASE, "baseline", .5), _row(GOOD, "templated", .6), _row(BAD, "llm", .5)
    r0 = Ratchet.from_baseline(baseline)
    assert r0.step(good)["verdict"] == "ACCEPT"
    assert r0.step(bad)["verdict"] == "REJECT" and r0.incumbent_sha == GOOD

    first = stratified_split(4, 8, 7)
    second = stratified_split(4, 8, 7)
    assert np.array_equal(first.open_in, second.open_in)
    r1 = Thresholdout.from_baseline(BASE, _artifact("base"), threshold=.01,
                                    split_seed=7, noise=lambda _position: (0., 0.))
    accepted = r1.step(good, _artifact("good"), 0)
    assert accepted["verdict"] == "ACCEPT" and not accepted["used_guard"]
    rejected = r1.step(bad, _artifact("bad"), 1)
    assert rejected["verdict"] == "REJECT" and r1.incumbent_sha == GOOD

    def verdict(row, incumbent, state, position):
        assert incumbent == BASE and state == {"t": 0} and position == 0
        state["t"] += 1
        return FakeDecision(_critic_row(row["sha"], row["cell"], incumbent))

    r2 = Critic(BASE, {"t": 0}, verdict)
    assert r2.step(good, 0)["verdict"] == "ACCEPT" and r2.incumbent_sha == GOOD and r2.state == {"t": 1}
    broken = Critic(BASE, {"t": 0}, lambda row, incumbent, state, position:
                    FakeDecision(_critic_row(row["sha"], row["cell"], incumbent, gate_g3="fail")))
    with pytest.raises(ValueError, match="every applicable"):
        broken.step(good, 0)

def test_replay_uses_same_order_for_every_rule_and_cell(tmp_path):
    rows = {(row["sha"], cell): row for cell in CELLS
            for row in (_row(BASE, "baseline", .5, cell),
                        _row(GOOD, "templated", .6, cell),
                        _row(BAD, "llm", .4, cell))}

    def critic_factory(cell, ordering, initial):
        def verdict(row, incumbent, state, position):
            state["t"] += 1
            return FakeDecision(_critic_row(row["sha"], row["cell"], incumbent))
        return Critic(initial, {"t": 0}, verdict)

    loader = lambda _cell, sha: _artifact("base" if sha == BASE else "good" if sha == GOOD else "bad")
    replays = [replay_cell(rows, cell, initial_sha=BASE, ordering_count=2, ordering_seed=8,
                           score_loader=loader, r1_threshold=.01, r1_split_seed=7,
                           noise_factory=lambda _cell, _ordering: lambda _position: (0., 0.),
                           critic_factory=critic_factory) for cell in CELLS]
    logs = combine_cells(*replays)
    assert len(logs) == 6 and all(len(items) == 8 for items in logs.values())
    write_decision_logs(tmp_path, logs, ordering_count=2)
    assert len(store.decision_digest(tmp_path)) == 64
    assert matched_orderings(rows, "clinc150/bert", 2, 8) == matched_orderings(rows, "clinc150/bert", 2, 8)
    with pytest.raises(ValueError, match="R0, R1 and R2"):
        write_decision_logs(tmp_path / "partial", {("R0", 0): []}, ordering_count=1)

def test_canonical_ledger_loader_accepts_existing_format():
    path = Path(__file__).resolve().parents[1] / "runs/ledger.jsonl"
    rows = load_ledger(path)
    assert len(rows) > 1000 and (GOOD, "clinc150/bert") not in rows
