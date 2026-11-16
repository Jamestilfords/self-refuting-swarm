"""The plain-model ECE job reads in-scope test logits and writes five seeds per arm."""

from __future__ import annotations

import json

import numpy as np
import pytest

from modal_jobs.plain_ece import _RUNS, collect_plain_ece, main, plain_model_ece
from skeptic.rules.common import TEST_N_IN
from skeptic.store import GuardError, SealedStore, UnsealPermit

PERMIT = UnsealPermit("a" * 64, "b" * 40)


def test_run_table_is_the_forty_fine_tunes():
    assert len(_RUNS) == 40
    covered = {}
    for group, cell, n_classes, arm, seed in _RUNS:
        covered.setdefault(cell, set()).add((group, arm, seed, n_classes))
    assert set(covered) == set(TEST_N_IN)
    for cell, runs in covered.items():
        assert len(runs) == 10
        assert {seed for _group, _arm, seed, _k in runs} == set(range(5))
        assert {arm for _group, arm, _seed, _k in runs} == {"ce", "edl"}
        classes = {k for _group, _arm, _seed, k in runs}
        assert classes == {150 if cell.startswith("clinc150") else 50}


def _write_cache(root, group, arm, seed, n_in, n_classes, *, labels=None):
    directory = root / group / f"{arm}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    logits = np.zeros((n_in, n_classes), dtype=np.float32)
    logits[:, 0] = 2.0
    if labels is None:
        labels = np.zeros(n_in, dtype=np.int64)
    np.savez_compressed(
        directory / "cache.npz",
        test__logits=logits,
        test__labels=np.asarray(labels),
    )


def test_collect_uses_in_scope_test_logits_only(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "modal_jobs.plain_ece._RUNS",
        (("clinc150_bert", "clinc150/bert", 3, "ce", 0),
         ("clinc150_bert", "clinc150/bert", 3, "edl", 0)),
    )
    monkeypatch.setitem(TEST_N_IN, "clinc150/bert", 4)
    _write_cache(tmp_path, "clinc150_bert", "ce", 0, 4, 3)
    _write_cache(tmp_path, "clinc150_bert", "edl", 0, 4, 3)
    store = SealedStore(tmp_path, tmp_path)
    store._permit = PERMIT
    payload = collect_plain_ece(store, PERMIT)
    assert payload["decision_commit"] == PERMIT.commit
    assert payload["decision_digest"] == PERMIT.digest
    assert list(payload["clinc150/bert"]) == ["ce", "edl"]
    with np.load(tmp_path / "clinc150_bert" / "ce_seed0" / "cache.npz") as cache:
        expected = plain_model_ece(cache["test__logits"], cache["test__labels"], "ce")
        assert "oos_test__labels" not in cache.files
    assert payload["clinc150/bert"]["ce"] == pytest.approx([expected], abs=1e-9)


def test_collect_rejects_a_short_test_split(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "modal_jobs.plain_ece._RUNS",
        (("banking77_bert", "banking77_oos/bert", 3, "ce", 1),),
    )
    monkeypatch.setitem(TEST_N_IN, "banking77_oos/bert", 8)
    _write_cache(tmp_path, "banking77_bert", "ce", 1, 4, 3)
    store = SealedStore(tmp_path, tmp_path)
    store._permit = PERMIT
    with pytest.raises(GuardError, match="wrong shape"):
        collect_plain_ece(store, PERMIT)


def test_main_writes_the_freeze_file_and_does_not_replace_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("skeptic.store.verify_published_freeze", lambda _repo: PERMIT)
    monkeypatch.setattr(
        "modal_jobs.plain_ece._RUNS",
        (("clinc150_bert", "clinc150/bert", 3, "ce", 0),),
    )
    monkeypatch.setitem(TEST_N_IN, "clinc150/bert", 4)
    sealed = tmp_path / "sealed"
    _write_cache(sealed, "clinc150_bert", "ce", 0, 4, 3)
    assert main(["--repo", str(tmp_path), "--sealed", str(sealed)]) == 0
    written = json.loads(capsys.readouterr().out)
    assert written["status"] == "written"
    payload = json.loads((sealed / "plain_ece.json").read_text(encoding="utf-8"))
    assert payload["decision_commit"] == PERMIT.commit
    with np.load(sealed / "clinc150_bert" / "ce_seed0" / "cache.npz") as cache:
        expected = plain_model_ece(cache["test__logits"], cache["test__labels"], "ce")
    assert payload["clinc150/bert"]["ce"] == pytest.approx([expected], abs=1e-9)
    assert main(["--repo", str(tmp_path), "--sealed", str(sealed)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "already_present"
    (sealed / "plain_ece.json").write_text(json.dumps({
        "decision_commit": "c" * 40,
        "decision_digest": "d" * 64,
        "clinc150/bert": {"ce": [0.0]},
    }))
    assert main(["--repo", str(tmp_path), "--sealed", str(sealed)]) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["status"] == "needs_sealed_logits"
    assert "provenance" in refused["error"]
