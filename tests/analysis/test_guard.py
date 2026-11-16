"""SealedSource refuses to read unless the decisions hash matches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skeptic.analysis.sources import GuardError, OpenSource, SealedSource
from skeptic.store import UnsealPermit


def _install_store(monkeypatch, runs, *, permit=None):
    import skeptic.store as store

    def verify(repo):
        hash_path = Path(repo) / "runs" / "DECISIONS.sha256"
        if permit is not None and hash_path.is_file():
            return permit
        if not hash_path.is_file():
            raise store.GuardError("runs/DECISIONS.sha256 is missing")
        raise store.GuardError("runs/DECISIONS.sha256 does not match the decision log")

    monkeypatch.setattr(store, "verify_published_freeze", verify)


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
    (runs / "DECISIONS.sha256").write_text("a" * 64 + "\n", encoding="utf-8")
    _install_store(monkeypatch, runs, permit=UnsealPermit("a" * 64, "b" * 40))
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
