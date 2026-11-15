"""Decision-log checks: pairing, the incumbent chain, and the sealed-volume guard."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from skeptic.analysis.decisions import load_decisions, write_decision_tree


def _ledger(shas, cell="clinc150/bert"):
    return pd.DataFrame([
        {"sha": sha, "cell": cell, "status": "ok", "stream": "templated", "family": "score", "canary": False}
        for sha in shas
    ])


def _rows(rule, ordering, sequence, accepts, cell="clinc150/bert"):
    rows = []
    incumbent = "energy_ce"
    for idx, sha in enumerate(sequence):
        take = sha in accepts
        rows.append({
            "rule": rule,
            "ordering": ordering,
            "sha": sha,
            "cell": cell,
            "idx": idx,
            "incumbent": incumbent,
            "accepted": take,
            "verdict": "ACCEPT" if take else "REJECT",
            "comparator": incumbent,
        })
        if take:
            incumbent = sha
    return rows


def test_load_accepts_a_matched_pair(tmp_path):
    shas = ["a", "b", "c"]
    frame = pd.DataFrame(
        _rows("R0", 7, shas, {"b"}) + _rows("R2", 7, shas, {"b", "c"})
    )
    write_decision_tree(frame, tmp_path / "decisions")
    loaded = load_decisions(tmp_path / "decisions", ledger=_ledger(shas))
    assert set(loaded["rule"]) == {"R0", "R2"}
    assert set(loaded["ordering"]) == {7}


def test_sequence_mismatch_on_ordering_7(tmp_path):
    shas = ["a", "b", "c"]
    frame = pd.DataFrame(
        _rows("R0", 7, ["a", "b", "c"], set()) + _rows("R2", 7, ["a", "c", "b"], set())
    )
    write_decision_tree(frame, tmp_path / "decisions")
    with pytest.raises(ValueError, match="sequence"):
        load_decisions(tmp_path / "decisions", ledger=_ledger(shas))


def test_broken_incumbent_chain(tmp_path):
    rows = _rows("R0", 0, ["a", "b"], {"a", "b"})
    rows[1]["incumbent"] = "energy_ce"
    write_decision_tree(pd.DataFrame(rows), tmp_path / "decisions")
    with pytest.raises(ValueError, match="incumbent"):
        load_decisions(tmp_path / "decisions", ledger=_ledger(["a", "b"]))


def test_r0_without_accepted_raises(tmp_path):
    rows = _rows("R0", 0, ["a"], set())
    del rows[0]["accepted"]
    folder = tmp_path / "decisions" / "R0"
    folder.mkdir(parents=True)
    (folder / "ord_000.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="accepted"):
        load_decisions(tmp_path / "decisions", ledger=_ledger(["a"]))


def test_r2_derives_accepted_from_verdict(tmp_path):
    body = [
        {"sha": "a", "cell": "clinc150/bert", "idx": 0, "incumbent": "energy_ce", "verdict": "ACCEPT"},
        {"sha": "b", "cell": "clinc150/bert", "idx": 1, "incumbent": "a", "verdict": "NO EFFECT"},
    ]
    folder = tmp_path / "decisions" / "R2"
    folder.mkdir(parents=True)
    (folder / "ord_000.jsonl").write_text(
        "\n".join(json.dumps(row) for row in body) + "\n", encoding="utf-8",
    )
    loaded = load_decisions(tmp_path / "decisions", ledger=_ledger(["a", "b"]))
    flags = dict(zip(loaded["sha"], loaded["accepted"]))
    assert flags == {"a": True, "b": False}


def test_directory_name_maps_to_canonical_rule(tmp_path):
    rows = _rows("critic", 0, ["a"], set())
    write_decision_tree(pd.DataFrame(rows), tmp_path / "decisions")
    loaded = load_decisions(tmp_path / "decisions", ledger=_ledger(["a"]))
    assert loaded.loc[0, "rule"] == "R2"
    assert loaded.loc[0, "rule_dir"] == "critic"


def test_idx_gap_raises(tmp_path):
    rows = _rows("R0", 0, ["a", "b"], set())
    rows[1]["idx"] = 2
    write_decision_tree(pd.DataFrame(rows), tmp_path / "decisions")
    with pytest.raises(ValueError, match="idx"):
        load_decisions(tmp_path / "decisions", ledger=_ledger(["a", "b"]))


def test_ledger_count_is_enforced(tmp_path):
    rows = _rows("R0", 0, ["a"], set())
    write_decision_tree(pd.DataFrame(rows), tmp_path / "decisions")
    with pytest.raises(ValueError, match="ledger"):
        load_decisions(tmp_path / "decisions", ledger=_ledger(["a", "b"]))
