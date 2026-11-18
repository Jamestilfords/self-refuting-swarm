"""Sealed-store/freeze checks independent of the E5 engine and Modal."""

import json
from pathlib import Path

import numpy as np
import pytest

from skeptic import store
from skeptic.rules.common import ACCEPT_GATES, ABLATION_SKIP, FROZEN_RULES, REFERENCE_BASELINES

CELLS = ("banking77_oos/bert", "banking77_oos/roberta", "clinc150/bert", "clinc150/roberta")
BASE, FIRST, SECOND = "a" * 12, "b" * 12, "c" * 12
HEAD = "d" * 40


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def decisions(tmp_path):
    (tmp_path / "PREREGISTRATION.md").write_text("**Status: FROZEN**\n", encoding="utf-8")
    files = []
    for rule in FROZEN_RULES:
        skip = ABLATION_SKIP.get(rule, frozenset())
        gates = {gate: ("skipped" if gate in skip else "pass") for gate in ACCEPT_GATES}
        for ordering in range(2):
            relative = f"runs/decisions/{rule}/ord_{ordering:03d}.jsonl"
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            sequence = (FIRST, SECOND) if ordering == 0 else (SECOND, FIRST)
            for cell in CELLS:
                incumbent = BASE
                for idx, sha in enumerate(sequence):
                    rows.append({"schema_version": 1, "rule": rule, "cell": cell,
                                 "ordering": ordering, "idx": idx, "sha": sha,
                                 "incumbent": incumbent, "verdict": "ACCEPT",
                                 "gates": gates})
                    incumbent = sha
            _write_rows(path, rows)
            files.append(relative)
    manifest = {"schema_version": 1, "rules": list(FROZEN_RULES), "orderings": 2,
                "cells": list(CELLS), "files": files,
                "proposals": {cell: [FIRST, SECOND] for cell in CELLS},
                "references": {cell: dict(REFERENCE_BASELINES) for cell in CELLS}}
    (tmp_path / store.MANIFEST_RELATIVE).write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return tmp_path


def test_freeze_is_deterministic_create_only_and_detects_byte_changes(decisions):
    expected = store.decision_digest(decisions)
    assert store.write_decision_hash(decisions) == expected
    assert store.verify_decision_hash(decisions) == expected
    with pytest.raises(store.GuardError, match="already exists"):
        store.write_decision_hash(decisions)
    target = decisions / "runs/decisions/R0/ord_000.jsonl"
    target.write_text(target.read_text().replace('"idx": 0', '"idx":  0'))
    with pytest.raises(store.GuardError, match="differs from the logs"):
        store.verify_decision_hash(decisions)


@pytest.mark.parametrize("registration", ["Status: DRAFT\n", "Status: FROZEN\nTODO: approve\n",
                                           "Status: Registered\n"])
def test_draft_registration_prevents_freeze(decisions, registration):
    (decisions / "PREREGISTRATION.md").write_text(registration)
    with pytest.raises(store.GuardError, match="draft"):
        store.write_decision_hash(decisions)
    assert not (decisions / store.HASH_RELATIVE).exists()


def test_registered_status_with_commit_is_accepted(tmp_path):
    (tmp_path / "PREREGISTRATION.md").write_text(
        "**Status: Registered. Commit " + "a" * 40 + ", 2026-11-15 09:52:44 UTC.**\n")
    store._check_registration(tmp_path)


@pytest.mark.parametrize("damage", [
    "missing_log", "missing_cell", "duplicate_position", "duplicate_proposal",
    "different_order", "broken_incumbent", "different_start", "failed_gate",
    "extra_log", "symlink",
])
def test_incomplete_or_inconsistent_decisions_cannot_freeze(decisions, damage):
    target = decisions / "runs/decisions/R2/ord_001.jsonl"
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    if damage == "missing_log":
        target.unlink()
    elif damage == "extra_log":
        (target.parent / "extra.jsonl").write_text("{}\n")
    elif damage == "symlink":
        other = decisions / "outside.jsonl"
        other.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(other)
    else:
        if damage == "missing_cell":
            rows = [row for row in rows if row["cell"] != CELLS[0]]
        elif damage == "duplicate_position":
            rows[1]["idx"] = 0
        elif damage == "duplicate_proposal":
            rows[1]["sha"] = rows[0]["sha"]
        elif damage == "different_order":
            rows[0]["sha"], rows[1]["sha"] = rows[1]["sha"], rows[0]["sha"]
            rows[1]["incumbent"] = rows[0]["sha"]
        elif damage == "broken_incumbent":
            rows[1]["incumbent"] = BASE
        elif damage == "different_start":
            rows[0]["incumbent"] = "e" * 12
        elif damage == "failed_gate":
            rows[0]["gates"]["G3"] = "fail"
        _write_rows(target, rows)
    with pytest.raises(store.GuardError):
        store.write_decision_hash(decisions)
    assert not (decisions / store.HASH_RELATIVE).exists()


def _published_git(repo, *, dirty=False, remote=HEAD, wrong_origin=False, missing=None, different=None):
    def run(_repo, *args):
        assert _repo == repo
        if args == ("status", "--porcelain"):
            return b" M file\n" if dirty else b""
        if args == ("remote", "get-url", "origin"):
            name = "other/repo" if wrong_origin else store.PUBLISHED_REPO
            return f"https://github.com/{name}.git\n".encode()
        if args == ("rev-parse", "HEAD"):
            return (HEAD + "\n").encode()
        if args == ("ls-remote", "origin", "refs/heads/main"):
            return f"{remote}\trefs/heads/main\n".encode()
        assert args[0] == "show"
        relative = args[1].split(":", 1)[1]
        if relative == missing:
            raise store.GuardError("Git proof failed: missing published log")
        if relative == different:
            return b"published bytes differ\n"
        return (repo / relative).read_bytes()
    return run


def test_published_proof_checks_every_frozen_file(decisions, monkeypatch):
    digest = store.write_decision_hash(decisions)
    monkeypatch.setattr(store, "_git_bytes", _published_git(decisions))
    assert store.verify_published_freeze(decisions) == store.UnsealPermit(digest, HEAD)


@pytest.mark.parametrize("problem", [
    {"dirty": True},
    {"remote": "f" * 40},
    {"wrong_origin": True},
    {"missing": "runs/decisions/R1/ord_000.jsonl"},
    {"different": "runs/decision_manifest.json"},
    {"different": "PREREGISTRATION.md"},
])
def test_clean_hash_alone_is_not_a_published_freeze(decisions, monkeypatch, problem):
    store.write_decision_hash(decisions)
    monkeypatch.setattr(store, "_git_bytes", _published_git(decisions, **problem))
    with pytest.raises(store.GuardError):
        store.verify_published_freeze(decisions)


def test_test_reads_require_proof_and_failed_recheck_revokes_permit(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    run = "clinc150_bert/ce_seed0"
    directory = root / run
    directory.mkdir(parents=True)
    np.savez(directory / "cache.npz", train__labels=np.array([0, 1]),
             val__labels=np.array([1, 0]), test__logits=np.ones((2, 3)),
             oos_test__logits=np.zeros((2, 3)))
    sealed = store.SealedStore(root, tmp_path)
    assert sealed.read_array(run, "val", "labels").tolist() == [1, 0]
    for split in ("test", "oos_test"):
        with pytest.raises(store.GuardError, match="sealed"):
            sealed.read_array(run, split, "logits")
    permit = store.UnsealPermit("a" * 64, HEAD)
    monkeypatch.setattr(store, "verify_published_freeze", lambda repo: permit)
    assert sealed.unseal() == permit
    assert sealed.read_array(run, "test", "logits").shape == (2, 3)
    with pytest.raises(store.GuardError, match="Unknown E2 run"):
        sealed.read_array("../elsewhere", "val", "labels")

    def denied(_repo):
        raise store.GuardError("changed freeze")

    monkeypatch.setattr(store, "verify_published_freeze", denied)
    with pytest.raises(store.GuardError, match="changed freeze"):
        sealed.unseal()
    with pytest.raises(store.GuardError, match="sealed"):
        sealed.read_array(run, "test", "logits")


def test_cache_directory_symlink_is_rejected(tmp_path):
    root = tmp_path / "cache"
    outside = tmp_path / "outside"
    root.mkdir()
    (outside / "ce_seed0").mkdir(parents=True)
    np.savez(outside / "ce_seed0/cache.npz", val__labels=np.array([0]))
    (root / "clinc150_bert").symlink_to(outside, target_is_directory=True)
    with pytest.raises(store.GuardError):
        store.SealedStore(root, tmp_path).read_array("clinc150_bert/ce_seed0", "val", "labels")


def test_reference_baseline_is_authorized_and_not_a_candidate(decisions, monkeypatch):
    from tests.test_seal import _publication

    store.write_decision_hash(decisions)
    monkeypatch.setattr(store, "_download", _publication(decisions))
    store.verify_remote_proposal(HEAD, CELLS[0], REFERENCE_BASELINES["energy_ce"])
    with pytest.raises(store.GuardError, match="not in the published"):
        store.verify_remote_proposal(HEAD, CELLS[0], "f" * 12)
    path = decisions / store.MANIFEST_RELATIVE
    manifest = json.loads(path.read_text())
    manifest["proposals"][CELLS[0]] = sorted(set(manifest["proposals"][CELLS[0]]) | {REFERENCE_BASELINES["energy_ce"]})
    path.write_text(json.dumps(manifest))
    with pytest.raises(store.GuardError, match="manifest"):
        store.decision_digest(decisions)


def test_remote_freeze_total_download_bound(decisions, monkeypatch):
    from tests.test_seal import _publication

    digest = store.write_decision_hash(decisions)
    monkeypatch.setattr(store, "_download", _publication(decisions))
    total = sum(path.stat().st_size for path in (decisions / "runs/decisions").rglob("*.jsonl"))
    monkeypatch.setattr(store, "MAX_DECISION_DOWNLOAD_BYTES", total)
    assert store.verify_remote_freeze(HEAD, digest) == store.UnsealPermit(digest, HEAD)
    monkeypatch.setattr(store, "MAX_DECISION_DOWNLOAD_BYTES", total - 1)
    with pytest.raises(store.GuardError, match="exceeds the download limit"):
        store.verify_remote_freeze(HEAD, digest)


def test_missing_ablation_or_extra_reference_cannot_freeze(decisions):
    variant = decisions / "runs/decisions/R2-noG3/ord_000.jsonl"
    variant.unlink()
    with pytest.raises(store.GuardError):
        store.write_decision_hash(decisions)
    variant.write_text((decisions / "runs/decisions/R2/ord_000.jsonl").read_text().replace('"rule": "R2"', '"rule": "R2-noG3"'))
    path = decisions / store.MANIFEST_RELATIVE
    manifest = json.loads(path.read_text())
    manifest["references"][CELLS[0]]["extra"] = "e" * 12
    path.write_text(json.dumps(manifest))
    with pytest.raises(store.GuardError, match="manifest"):
        store.write_decision_hash(decisions)


def test_cli_hash_and_verify_only_touch_freeze_files(decisions, capsys):
    assert store.main(["freeze", "--repo", str(decisions)]) == 0
    digest = capsys.readouterr().out.strip()
    assert len(digest) == 64
    assert store.main(["verify", "--repo", str(decisions)]) == 0
    assert capsys.readouterr().out.strip() == digest


@pytest.mark.parametrize("changed", [False, True])
def test_registered_freeze_identity_survives_only_runner_changes(decisions, monkeypatch, changed):
    digest = store.write_decision_hash(decisions)
    anchor = "e" * 40
    monkeypatch.setattr(store, "REGISTERED_FREEZES", {digest: anchor})
    original = _published_git(decisions)
    def git(repo, *args):
        if args[0] == "rev-parse" and ":" in args[1]:
            commit, relative = args[1].split(":", 1)
            return ("different" if changed and commit == anchor and relative == "runs/decisions"
                    else "same").encode()
        return original(repo, *args)
    monkeypatch.setattr(store, "_git_bytes", git)
    if changed:
        with pytest.raises(store.GuardError, match="registered commit"):
            store.verify_published_freeze(decisions)
    else:
        assert store.verify_published_freeze(decisions) == store.UnsealPermit(digest, anchor)


def test_registered_remote_freeze_remains_pinned_after_runner_update(decisions, monkeypatch):
    from tests.test_seal import _publication
    digest = store.write_decision_hash(decisions)
    monkeypatch.setattr(store, "REGISTERED_FREEZES", {digest: HEAD})
    monkeypatch.setattr(store, "_download", _publication(decisions, head="e" * 40))
    assert store.verify_remote_freeze(HEAD, digest) == store.UnsealPermit(digest, HEAD)
