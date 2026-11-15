"""SealedSource refuses to read unless the decisions hash matches."""

from __future__ import annotations

import numpy as np
import pytest

from skeptic.analysis.decisions import decisions_tree_sha256
from skeptic.analysis.sources import GuardError, OpenSource, SealedSource


def _install_store(monkeypatch, runs):
    import skeptic.store as store

    def is_unsealed():
        hash_path = runs / "DECISIONS.sha256"
        if not hash_path.is_file():
            raise GuardError("runs/DECISIONS.sha256 is missing")
        recorded = hash_path.read_text(encoding="utf-8").strip().split()[0]
        actual = decisions_tree_sha256(runs / "decisions")
        if recorded != actual:
            raise GuardError("runs/DECISIONS.sha256 does not match the decision log")
        return True

    monkeypatch.setattr(store, "is_unsealed", is_unsealed, raising=False)


def _write_scores(runs):
    folder = runs / "sealed" / "clinc150" / "bert" / "energy_ce"
    folder.mkdir(parents=True)
    np.save(folder / "seed0.npy", np.arange(10, dtype=np.float64))
    decision = runs / "decisions" / "R0"
    decision.mkdir(parents=True)
    (decision / "ord_000.jsonl").write_text("{}\n", encoding="utf-8")


def test_sealed_source_refuses_a_missing_hash(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    _write_scores(runs)
    _install_store(monkeypatch, runs)
    with pytest.raises(GuardError):
        SealedSource(runs / "sealed", {"clinc150/bert": 4})


def test_sealed_source_refuses_a_wrong_hash(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    _write_scores(runs)
    (runs / "DECISIONS.sha256").write_text("deadbeef\n", encoding="utf-8")
    _install_store(monkeypatch, runs)
    with pytest.raises(GuardError):
        SealedSource(runs / "sealed", {"clinc150/bert": 4})


def test_sealed_source_reads_after_a_matching_hash(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    _write_scores(runs)
    digest = decisions_tree_sha256(runs / "decisions")
    (runs / "DECISIONS.sha256").write_text(digest + "\n", encoding="utf-8")
    _install_store(monkeypatch, runs)
    source = SealedSource(runs / "sealed", {"clinc150/bert": 4})
    ident, oos = source.get("clinc150/bert", "energy_ce")
    assert ident[0].tolist() == [0, 1, 2, 3]
    assert oos[0].tolist() == [4, 5, 6, 7, 8, 9]
    log = (runs / "guard.log").read_text(encoding="utf-8")
    assert 'purpose="truth"' in log
    assert 'cell="clinc150/bert"' in log
    assert 'name="energy_ce"' in log


def test_open_source_splits_validation_arrays(tmp_path):
    folder = tmp_path / "open" / "clinc150" / "bert" / "energy_ce"
    folder.mkdir(parents=True)
    np.save(folder / "seed0.npy", np.arange(6, dtype=np.float64))
    source = OpenSource(tmp_path / "open", {"clinc150/bert": 2})
    assert source.kind == "val"
    ident, oos = source.get("clinc150/bert", "energy_ce")
    assert ident[0].tolist() == [0, 1]
    assert oos[0].tolist() == [2, 3, 4, 5]
    assert not (tmp_path / "open" / "guard.log").exists()
