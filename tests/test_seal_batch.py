"""Batch planning/orchestration tests: synthetic proposals and mocked Modal/GitHub only."""
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from modal_jobs import seal
from skeptic import api, store
from skeptic.rules.common import FROZEN_RULES, REFERENCE_BASELINES
from skeptic.streams.canaries import CANARIES
from tests.test_store import CELLS, HEAD, FIRST, decisions
from tests.test_seal import _publication


@pytest.fixture
def batch_repo(tmp_path, monkeypatch):
    proposals = tmp_path / "proposals"
    proposals.mkdir()
    shas = []
    for candidate in CANARIES[:2]:
        text = candidate["text"]
        sha = api.proposal_sha(text)
        (proposals / f"{sha}.py").write_text(text)
        shas.append(sha)
    repo = Path(__file__).resolve().parents[1]
    for sha in REFERENCE_BASELINES.values():
        text = (repo / "proposals" / f"{sha}.py").read_text(encoding="utf-8")
        (proposals / f"{sha}.py").write_text(text, encoding="utf-8")
    manifest = {"schema_version": 1, "rules": list(FROZEN_RULES), "orderings": 1,
                "cells": list(CELLS),
                "files": [f"runs/decisions/{rule}/ord_000.jsonl" for rule in FROZEN_RULES],
                "proposals": {cell: sorted(shas) for cell in CELLS},
                "references": {cell: dict(REFERENCE_BASELINES) for cell in CELLS}}
    (tmp_path / "runs").mkdir()
    (tmp_path / store.MANIFEST_RELATIVE).write_text(json.dumps(manifest))
    monkeypatch.setattr(seal, "REPO", tmp_path)
    return tmp_path


def test_batch_plan_has_all_frozen_cells_and_no_side_effects(batch_repo, monkeypatch, capsys):
    monkeypatch.setattr(seal, "verify_published_freeze", lambda *a: pytest.fail("network in plan"))
    monkeypatch.setattr(seal.seal_app, "run", lambda **kw: pytest.fail("Modal in plan"))
    assert seal.main(["plan", "--all", "--batch-size", "3"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["job_count"] == 28 and plan["batches"] == 10
    assert plan["max_sandbox_executions"] == 280 and plan["freeze_verified"] is False
    assert {j["cell"] for j in plan["jobs"]} == set(CELLS)
    assert all("text" not in j for j in plan["jobs"])
    assert not (batch_repo / "runs/seal").exists()
    assert seal.main(["plan", "--all", "--dataset", "banking77", "--backbone", "roberta"]) == 0
    filtered = json.loads(capsys.readouterr().out)
    assert filtered["job_count"] == 7
    assert all(j["dataset"] == "banking77_oos" for j in filtered["jobs"])


def test_single_plan_retains_cli_and_normalizes_banking(batch_repo, capsys):
    proposal = next((batch_repo / "proposals").glob("*.py"))
    assert seal.main(["plan", "--proposal", str(proposal), "--dataset", "banking77", "--backbone", "bert"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["sha"] == proposal.stem and plan["cell"] == "banking77_oos/bert"
    assert plan["dataset"] == "banking77_oos"


@pytest.mark.parametrize("args", [["--all", "--batch-size", "0"], ["--all", "--batch-size", "26"],
                                  ["--all", "--proposal", "anything.py"], ["--proposal", "anything.py"]])
def test_invalid_selection_stops_before_modal(batch_repo, args):
    with pytest.raises(SystemExit) as exc:
        seal.main(["plan", *args])
    assert exc.value.code == 2


@pytest.mark.parametrize("damage", ["missing_manifest", "missing_source", "changed_source", "symlink_source"])
def test_plan_fails_closed_on_incomplete_or_changed_inputs(batch_repo, damage):
    path = next((batch_repo / "proposals").glob("*.py"))
    if damage == "missing_manifest":
        (batch_repo / store.MANIFEST_RELATIVE).unlink()
    elif damage == "missing_source":
        path.unlink()
    elif damage == "changed_source":
        path.write_text(path.read_text() + "# changed\n")
    else:
        copy = batch_repo / "other.py"
        copy.write_text(path.read_text())
        path.unlink()
        path.symlink_to(copy)
    with pytest.raises(SystemExit) as exc:
        seal.main(["plan", "--all"])
    assert exc.value.code == 2


def test_remote_batch_membership_checks_every_target_with_one_manifest(decisions, monkeypatch):
    download = _publication(decisions)
    calls = []
    def fetch(url, limit):
        calls.append(url)
        return download(url, limit)
    monkeypatch.setattr(store, "_download", fetch)
    store.verify_remote_proposals(HEAD, [(cell, FIRST) for cell in CELLS])
    assert len(calls) == 1
    with pytest.raises(store.GuardError, match="not in the published"):
        store.verify_remote_proposals(HEAD, [(CELLS[0], FIRST), (CELLS[1], "f" * 12)])


def _remote_fixture(monkeypatch):
    events = []
    monkeypatch.setattr(seal, "sealed_volume", SimpleNamespace(reload=lambda: events.append("reload")))
    def unseal(*args):
        events.append("freeze")
    def membership(commit, targets):
        events.append("membership")
    monkeypatch.setattr(seal.SealedStore, "unseal_remote", unseal)
    monkeypatch.setattr(seal, "verify_remote_proposals", membership)
    return events


def test_remote_batch_verifies_before_scoring_and_continues_after_failure(batch_repo, monkeypatch):
    events = _remote_fixture(monkeypatch)
    jobs = seal.plan_jobs(None, "banking77", "bert")
    def score(text, specs, *args):
        assert events[:3] == ["reload", "freeze", "membership"]
        assert all(s.dataset == "banking77_oos" for s in specs)
        events.append("score")
        if len(events) == 4:
            raise RuntimeError("PRIVATE TEST DATA MUST NOT LEAK")
        return {"sha": api.proposal_sha(text), "cell": "banking77_oos/bert", "status": "already_sealed"}
    monkeypatch.setattr(seal, "_score_authorized", score)
    with pytest.warns(UserWarning, match="executing locally"):
        result = seal.score_sealed_batch.local(jobs, HEAD, "a" * 64, "im-test")
    assert [r["status"] for r in result] == ["failed", *["already_sealed"] * 6]
    assert "PRIVATE" not in json.dumps(result)
    assert events.count("freeze") == 1 and events.count("membership") == 1


@pytest.mark.parametrize("failure", ["freeze", "membership"])
def test_batch_denied_before_any_scoring(batch_repo, monkeypatch, failure):
    _remote_fixture(monkeypatch)
    def reject(*args):
        raise store.GuardError("not authorized")
    if failure == "freeze":
        monkeypatch.setattr(seal.SealedStore, "unseal_remote", reject)
    else:
        monkeypatch.setattr(seal, "verify_remote_proposals", reject)
    monkeypatch.setattr(seal, "_score_authorized", lambda *a: pytest.fail("score before authorization"))
    with pytest.warns(UserWarning, match="executing locally"), pytest.raises(store.GuardError):
        seal.score_sealed_batch.local(seal.plan_jobs(None, "banking77", "bert"), HEAD, "a" * 64, "im-test")


def _cli_fixture(monkeypatch, mapping):
    receipt = {}
    monkeypatch.setattr(seal.modal.Dict, "ephemeral", lambda **kw: nullcontext(receipt))
    monkeypatch.setattr(seal, "verify_published_freeze", lambda *a: store.UnsealPermit("a" * 64, HEAD))
    monkeypatch.setattr(seal, "seal_app", SimpleNamespace(run=lambda **kw: nullcontext()))
    monkeypatch.setattr(seal.modal, "enable_output", nullcontext)
    monkeypatch.setattr(seal, "sandbox_image", SimpleNamespace(object_id="im-test"))
    monkeypatch.setattr(seal, "score_sealed_batch", SimpleNamespace(starmap=mapping))


def test_cli_chunks_reports_failures_and_retries_all_remote_jobs(batch_repo, monkeypatch, capsys):
    calls = []
    def mapping(inputs, **kwargs):
        assert kwargs == {"order_outputs": True, "return_exceptions": True}
        calls.append(inputs)
        for i, (jobs, commit, digest, image, authorization) in enumerate(inputs):
            assert len(jobs) <= 3 and commit == HEAD and image == "im-test"
            assert authorization["freeze"]["commit"] == commit
            if len(calls) == 1 and i == 1:
                yield RuntimeError("PRIVATE REMOTE ERROR")
            else:
                yield [{"sha": j["sha"], "cell": j["cell"],
                        "status": "sealed" if len(calls) == 1 else "already_sealed"} for j in jobs]
    _cli_fixture(monkeypatch, mapping)
    args = ["run", "--all", "--batch-size", "3"]
    assert seal.main(args) == 1
    report = batch_repo / "runs/seal/status.jsonl"
    rows = [json.loads(l) for l in report.read_text().splitlines()]
    assert len(rows) == 28 and sum(r["status"] == "failed" for r in rows) == 3
    assert all(r["decision_commit"] == HEAD for r in rows)
    assert "PRIVATE" not in capsys.readouterr().out + report.read_text()
    assert seal.main(args) == 0
    assert calls[0] == calls[1]  # Local success rows never replace remote provenance checks.
    assert len(report.read_text().splitlines()) == 56


@pytest.mark.parametrize("damage", ["extra_metrics", "wrong_identity", "missing_result", "failed_status"])
def test_cli_sanitizes_or_rejects_bad_remote_results(batch_repo, monkeypatch, capsys, damage):
    def mapping(inputs, **kwargs):
        for jobs, *_ in inputs:
            rows = [{"sha": j["sha"], "cell": j["cell"], "status": "sealed"} for j in jobs]
            if damage == "extra_metrics":
                rows[0]["test_auroc"] = "PRIVATE"
            elif damage == "wrong_identity":
                rows[0]["sha"] = "PRIVATE"
            elif damage == "failed_status":
                rows[0].update(status="failed", error_type="PRIVATE")
            else:
                rows.pop()
            yield rows
    _cli_fixture(monkeypatch, mapping)
    assert seal.main(["run", "--all"]) == 1
    assert "PRIVATE" not in capsys.readouterr().out + (batch_repo / "runs/seal/status.jsonl").read_text()


def test_cli_guard_failure_never_starts_modal_or_creates_report(batch_repo, monkeypatch):
    def reject(*args):
        raise store.GuardError("freeze missing")
    monkeypatch.setattr(seal, "verify_published_freeze", reject)
    monkeypatch.setattr(seal.seal_app, "run", lambda **kw: pytest.fail("started before freeze verified"))
    with pytest.raises(store.GuardError):
        seal.main(["run", "--all"])
    assert not (batch_repo / "runs/seal").exists()


def test_shared_authorization_rejects_mismatch_before_test_reads(tmp_path):
    sealed = store.SealedStore(tmp_path, tmp_path)
    target = ("clinc150/bert", "a" * 12)
    receipt = {"freeze": {"commit": HEAD, "digest": "b" * 64, "targets": [target]}}
    for commit, digest, targets in (("e" * 40, "b" * 64, [target]),
                                    (HEAD, "c" * 64, [target]),
                                    (HEAD, "b" * 64, [(target[0], "f" * 12)])):
        with pytest.raises(store.GuardError):
            seal._authorize_batch(sealed, receipt, commit, digest, targets)
        with pytest.raises(store.GuardError):
            sealed.guard_read("test")
    seal._authorize_batch(sealed, receipt, HEAD, "b" * 64, [target])
    sealed.guard_read("test")


def test_authorized_batch_does_not_download_github(batch_repo, monkeypatch):
    monkeypatch.setattr(seal, "sealed_volume", SimpleNamespace(reload=lambda: None))
    monkeypatch.setattr(store, "_download", lambda *a: pytest.fail("batch downloaded GitHub"))
    jobs = seal.plan_jobs(None, "banking77", "bert")
    receipt = {"freeze": {"commit": HEAD, "digest": "a" * 64,
                           "targets": [(j["cell"], j["sha"]) for j in jobs]}}
    def score(text, specs, sealed, *args):
        sealed.guard_read("test")
        return {"sha": api.proposal_sha(text), "cell": "banking77_oos/bert", "status": "already_sealed"}
    monkeypatch.setattr(seal, "_score_authorized", score)
    with pytest.warns(UserWarning, match="executing locally"):
        result = seal.score_sealed_batch.local(jobs, HEAD, "a" * 64, "im-test", receipt)
    assert len(result) == len(jobs) and all(r["status"] == "already_sealed" for r in result)
