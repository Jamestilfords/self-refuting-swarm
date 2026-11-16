"""Offline tests for the sealed plain-model ECE job. No Modal calls."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from modal_jobs import plain_ece
from skeptic.rules.common import TEST_N_IN
from skeptic.score_artifacts import CELLS
from skeptic.store import GuardError, SealedStore, UnsealPermit

PERMIT = UnsealPermit("a" * 64, "b" * 40)
CE_ECE = 0.449151754485
EDL_ECE = 0.449151572702


def test_plain_model_ece_matches_precomputed_validation_arrays(tmp_path):
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(12, 4))
    labels = rng.integers(0, 4, size=12)
    for arm in ("ce", "edl"):
        directory = tmp_path / "clinc150_bert" / f"{arm}_seed0"
        directory.mkdir(parents=True)
        np.savez_compressed(directory / "cache.npz", val__logits=logits, val__labels=labels)
    with np.load(tmp_path / "clinc150_bert" / "ce_seed0" / "cache.npz") as cache:
        assert plain_ece.plain_model_ece(cache["val__logits"], cache["val__labels"], "ce") == pytest.approx(CE_ECE, abs=1e-9)
    with np.load(tmp_path / "clinc150_bert" / "edl_seed0" / "cache.npz") as cache:
        assert plain_ece.plain_model_ece(cache["val__logits"], cache["val__labels"], "edl") == pytest.approx(EDL_ECE, abs=1e-9)


def test_ledger_ece_plain_when_the_open_cache_is_present():
    cache = Path("runs/open-caches/banking77_bert/ce_seed0/cache.npz")
    if not cache.is_file():
        pytest.skip("open caches are not on this machine")
    with np.load(cache) as saved:
        value = plain_ece.plain_model_ece(saved["val__logits"], saved["val__labels"], "ce")
    print(json.dumps({"cell": "banking77_oos/bert", "seed": 0, "ece_plain": value}))


def _mount(monkeypatch, root: Path, *, permit=True):
    events = []
    monkeypatch.setattr(plain_ece, "SEALED_MOUNT", str(root))
    monkeypatch.setattr(plain_ece, "sealed_volume", SimpleNamespace(
        reload=lambda: events.append("reload"),
        commit=lambda: events.append("commit"),
    ))

    def unseal(self, commit, digest):
        events.append("unseal")
        if permit:
            self._permit = UnsealPermit(digest, commit)
            return self._permit
        self._permit = None
        return None

    monkeypatch.setattr(SealedStore, "unseal_remote", unseal)
    return events


def _cache(root: Path, n: int, k: int = 3):
    logits = np.zeros((n, k), dtype=np.float32)
    logits[:, 0] = 1.0
    labels = np.zeros(n, dtype=np.int64)
    for group in CELLS.values():
        for arm in ("ce", "edl"):
            for seed in range(5):
                directory = root / group / f"{arm}_seed{seed}"
                directory.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(directory / "cache.npz", test__logits=logits, test__labels=labels)


def _contains_float(value) -> bool:
    if isinstance(value, float):
        return True
    if isinstance(value, dict):
        return any(_contains_float(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_float(item) for item in value)
    return False


def test_compute_plain_ece_writes_five_seeds_and_returns_no_floats(tmp_path, monkeypatch):
    events = _mount(monkeypatch, tmp_path)
    for cell in TEST_N_IN:
        monkeypatch.setitem(TEST_N_IN, cell, 4)
    _cache(tmp_path, 4)
    with pytest.warns(UserWarning, match="executing locally"):
        result = plain_ece.compute_plain_ece.local(PERMIT.commit, PERMIT.digest)
    assert events[:2] == ["reload", "unseal"]
    assert events[-1] == "commit"
    assert result == {"status": "written", "cells": 4, "values": 40}
    assert not _contains_float(result)
    payload = json.loads((tmp_path / "plain_ece.json").read_text(encoding="utf-8"))
    assert payload["decision_commit"] == PERMIT.commit
    assert payload["decision_digest"] == PERMIT.digest
    assert set(payload) == {"decision_commit", "decision_digest", *CELLS}
    for cell in CELLS:
        assert set(payload[cell]) == {"ce", "edl"}
        assert len(payload[cell]["ce"]) == 5 and len(payload[cell]["edl"]) == 5
        assert all(np.isfinite(value) for value in payload[cell]["ce"])


def test_compute_plain_ece_refuses_without_unseal_or_on_a_bad_cache(tmp_path, monkeypatch):
    _mount(monkeypatch, tmp_path, permit=False)
    with pytest.warns(UserWarning, match="executing locally"):
        with pytest.raises(GuardError, match="sealed"):
            plain_ece.compute_plain_ece.local(PERMIT.commit, PERMIT.digest)
    assert not (tmp_path / "plain_ece.json").exists()

    _mount(monkeypatch, tmp_path)
    (tmp_path / "plain_ece.json").write_text(json.dumps({
        "decision_commit": "c" * 40, "decision_digest": "d" * 64,
    }))
    with pytest.warns(UserWarning, match="executing locally"):
        with pytest.raises(GuardError, match="provenance"):
            plain_ece.compute_plain_ece.local(PERMIT.commit, PERMIT.digest)
    assert "c" * 40 in (tmp_path / "plain_ece.json").read_text(encoding="utf-8")

    (tmp_path / "plain_ece.json").unlink()
    directory = tmp_path / "clinc150_bert" / "ce_seed0"
    directory.mkdir(parents=True)
    np.savez_compressed(
        directory / "cache.npz",
        test__logits=np.zeros((3, 3), dtype=np.float32),
        test__labels=np.zeros(3, dtype=np.int64),
    )
    with pytest.warns(UserWarning, match="executing locally"):
        with pytest.raises(GuardError, match="wrong length"):
            plain_ece.compute_plain_ece.local(PERMIT.commit, PERMIT.digest)
    assert not (tmp_path / "plain_ece.json").exists()


def test_main_verifies_the_freeze_before_modal(monkeypatch):
    order = []

    def verify(_repo):
        order.append("verify")
        raise GuardError("no freeze")

    monkeypatch.setattr("skeptic.store.verify_published_freeze", verify)
    monkeypatch.setattr(plain_ece.ece_app, "run", lambda **_kwargs: pytest.fail("Modal"))
    assert plain_ece.main(["--repo", "."]) == 1
    assert order == ["verify"]

    def verified(_repo):
        order.append("verify")
        return PERMIT

    class Boom:
        def __enter__(self):
            order.append("modal")
            raise RuntimeError("no modal")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr("skeptic.store.verify_published_freeze", verified)
    monkeypatch.setattr(plain_ece.ece_app, "run", lambda **_kwargs: Boom())
    with pytest.raises(RuntimeError, match="no modal"):
        plain_ece.main(["--repo", "."])
    assert order == ["verify", "verify", "modal"]
