"""E3 protocol, isolation configuration, and ledger checks; no Modal jobs."""

import io
import json
from pathlib import Path

import numpy as np
import pytest

from modal_jobs import sandbox as job
from modal_jobs.train40 import RunSpec, _fingerprint, _recipe
from skeptic import api, ledger
from skeptic.sandbox import pack_input, unpack_input, unpack_output, worker
from skeptic.streams.canaries import CANARIES


def _arrays(n, classes=150, *, first=0):
    logits = np.zeros((n, classes), np.float32)
    logits[:, 0] = np.arange(first, first + n) + 1
    return {"logits": logits, "feats": np.ones((n, 256), np.float32),
            "pooled": np.ones((n, 768), np.float32)}


def _open_cache(root: Path, spec: RunSpec):
    directory = root / spec.name
    directory.mkdir(parents=True)
    k = 150 if spec.dataset == "clinc150" else 50
    train, val, oos = _arrays(6, k), _arrays(4, k, first=10), _arrays(3, k, first=30)
    arrays = {f"{split}__{key}": value for split, data in
              (("train", train), ("val", val), ("oos_val", oos)) for key, value in data.items()}
    arrays["train__labels"] = np.zeros(6, np.int64)
    np.savez_compressed(directory / "cache.npz", **arrays)
    recipe = _recipe(spec)
    (directory / "manifest.json").write_text(json.dumps({"run": recipe,
        "recipe_hash": _fingerprint(recipe)}), encoding="utf-8")
    return directory


@pytest.mark.parametrize("dataset", ["banking77", "banking77_oos"])
@pytest.mark.parametrize("backbone", ["bert", "roberta"])
@pytest.mark.parametrize("arm", ["ce", "edl"])
def test_banking_alias_uses_original_training_recipe_and_cache_path(tmp_path, dataset, backbone, arm):
    canonical = RunSpec("banking77_oos", backbone, arm, 0)
    spec = job._spec(dataset, backbone, arm, 0)
    assert spec == canonical and spec.name == f"banking77_{backbone}/{arm}_seed0"
    _open_cache(tmp_path, canonical)
    full, _, _, _, k = job.prepare_cell(tmp_path, spec)
    assert k == 50 and unpack_input(full).eval.logits.shape[1] == 50


def test_prepare_cell_sends_anonymous_eval_and_rejects_extra_arrays(tmp_path):
    spec = RunSpec("clinc150", "bert", "ce", 0)
    directory = _open_cache(tmp_path, spec)
    full, subset, membership, subset_idx, k = job.prepare_cell(tmp_path, spec)
    ctx = unpack_input(full)
    assert len(ctx.train) == 6 and len(ctx.eval) == 7 and k == 150
    assert ctx.eval.labels is None
    assert len(unpack_input(subset).eval) == len(subset_idx) == 2
    assert membership.sum() == 3
    assert np.all(ctx.eval.logits[membership, 0] >= 31)
    assert np.all(ctx.eval.logits[~membership, 0] <= 14)
    assert b"oos_val" not in full and b"val__labels" not in full

    with np.load(directory / "cache.npz", allow_pickle=False) as cache:
        arrays = {name: cache[name] for name in cache.files}
    arrays["test__labels"] = np.array([0])
    np.savez_compressed(directory / "cache.npz", **arrays)
    with pytest.raises(RuntimeError, match="Unexpected or missing arrays"):
        job.prepare_cell(tmp_path, spec)


def test_worker_scores_known_proposal_and_controller_checks_output(tmp_path):
    text = CANARIES[0]["text"]
    path = tmp_path / f"{api.proposal_sha(text)}.py"
    path.write_text(text, encoding="utf-8")
    train, eval_data = _arrays(6), _arrays(7, first=10)
    train["labels"] = np.zeros(6, np.int64)
    input_path = tmp_path / "input.npz"
    input_path.write_bytes(pack_input(train, eval_data))
    output_path, status_path = tmp_path / "output.npz", tmp_path / "status.json"
    worker(str(path), str(input_path), str(output_path), str(status_path))
    assert json.loads(status_path.read_text()) == {"status": "ok"}
    scores, runtime = unpack_output(output_path.read_bytes(), 7, 150)
    assert np.allclose(scores, 2 * api.Helpers.energy(eval_data["logits"]) + 1)
    assert runtime >= 0

    with np.load(output_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    arrays["test_scores"] = np.zeros(7)
    altered = io.BytesIO()
    np.savez_compressed(altered, **arrays)
    with pytest.raises(api.Malformed, match="sandbox_error"):
        unpack_output(altered.getvalue(), 7, 150)


def test_leak_canary_is_rejected_without_running_or_using_modal(tmp_path):
    text = CANARIES[-1]["text"]
    path = tmp_path / f"{api.proposal_sha(text)}.py"
    path.write_text(text, encoding="utf-8")
    _, row = job.preflight(path, RunSpec("clinc150", "bert", "ce", 0), "canary")
    assert row["status"] == "io" and row["validation_auroc"] is None
    options = job.sandbox_options(object(), object())
    assert options["block_network"] is True and options["runtime"] == "gvisor"
    assert options["volumes"] == {} and options["secrets"] == []
    assert options["include_oidc_identity_token"] is False
    assert options["env"]["MODAL_TOKEN_SECRET"] is None


def test_ledger_is_append_only_and_cannot_record_raw_or_test_scores(tmp_path):
    text = CANARIES[-1]["text"]
    path = tmp_path / f"{api.proposal_sha(text)}.py"
    path.write_text(text, encoding="utf-8")
    _, row = job.preflight(path, RunSpec("clinc150", "bert", "ce", 0), "canary")
    path = tmp_path / "ledger.jsonl"
    assert ledger.append_row(path, row) is True
    assert ledger.append_row(path, row) is False
    assert ledger.read_rows(path) == [row]
    with pytest.raises(ValueError, match="public E3 fields"):
        ledger.append_row(path, {**row, "test_scores": [0.5]})
    with pytest.raises(ValueError, match="Conflicting row"):
        ledger.append_row(path, {**row, "status": "timeout"})


def test_subset_worker_allows_constant_subset(tmp_path):
    text = """FAMILY = "logit"
ARM = "ce"
RATIONALE = "First logit"
SOURCE = ""
PARENTS = []
MEMORY_USED = []

def score(ctx):
    return ctx.eval.logits[:, 0]
"""
    path = tmp_path / f"{api.proposal_sha(text)}.py"
    path.write_text(text, encoding="utf-8")
    train, eval_data = _arrays(6), _arrays(2)
    train["labels"] = np.zeros(6, np.int64)
    eval_data["logits"][:, 0] = 1
    input_path = tmp_path / "input.npz"
    input_path.write_bytes(pack_input(train, eval_data))
    output_path, status_path = tmp_path / "output.npz", tmp_path / "status.json"
    worker(str(path), str(input_path), str(output_path), str(status_path), subset=True)
    assert json.loads(status_path.read_text()) == {"status": "ok"}
    scores, _ = unpack_output(output_path.read_bytes(), 2, 150, require_variation=False)
    assert np.array_equal(scores, np.ones(2))


def test_trusted_wrapper_compares_fresh_subset_and_reports_only_auroc(tmp_path, monkeypatch):
    spec = RunSpec("clinc150", "bert", "ce", 0)
    _open_cache(tmp_path, spec)
    monkeypatch.setattr(job, "OPEN_MOUNT", str(tmp_path))
    monkeypatch.setattr(job, "SEALED_MOUNT", str(tmp_path / "sealed"))
    monkeypatch.setattr(job.modal.App, "lookup", staticmethod(lambda *args, **kwargs: object()))
    monkeypatch.setattr(job.modal.Image, "from_id", staticmethod(lambda *args, **kwargs: object()))
    text = CANARIES[0]["text"]

    def score(_owner, _image, _text, _sha, payload, n, k, *, subset=False):
        ctx = unpack_input(payload)
        assert len(ctx.eval) == n and k == 150
        return np.asarray(ctx.eval.logits[:, 0], np.float64), 0.02

    monkeypatch.setattr(job, "_score_in_sandbox", score)
    with pytest.warns(UserWarning, match="executing locally"):
        row = job.score_one.local(text, "clinc150", "bert", "ce", 0, "im-example", "canary")
    assert row["status"] == "ok" and row["validation_auroc"] == 1.0
    assert set(row) == ledger.FIELDS and row["n_id"] == 4 and row["n_oos"] == 3

    def inconsistent(_owner, _image, _text, _sha, payload, n, k, *, subset=False):
        scores, seconds = score(_owner, _image, _text, _sha, payload, n, k, subset=subset)
        return scores + (1 if subset else 0), seconds

    monkeypatch.setattr(job, "_score_in_sandbox", inconsistent)
    with pytest.warns(UserWarning, match="executing locally"):
        rejected = job.score_one.local(text, "clinc150", "bert", "ce", 0, "im-example", "canary")
    assert rejected["status"] == "subset_dependent" and rejected["validation_auroc"] is None


@pytest.mark.parametrize("times_out", [False, True])
def test_controller_passes_extended_deadlines_and_always_terminates(monkeypatch, times_out):
    from types import SimpleNamespace

    created, executed, terminated = [], [], []
    output = io.BytesIO()
    np.savez_compressed(output, scores=np.arange(7, dtype=float), probs=np.empty(0), runtime_s=45.0)
    files = {"/tmp/status.json": b'{"status":"ok"}', "/tmp/output.npz": output.getvalue()}

    def wait():
        if times_out:
            raise job.modal.exception.ExecTimeoutError("synthetic timeout")
        return 0

    def execute(*args, **kwargs):
        executed.append(kwargs)
        return SimpleNamespace(wait=wait)

    fake = SimpleNamespace(
        filesystem=SimpleNamespace(write_text=lambda *a: None, write_bytes=lambda *a: None,
                                   stat=lambda path: SimpleNamespace(size=len(files[path])),
                                   read_bytes=lambda path: files[path]),
        exec=execute, terminate=lambda: terminated.append(True))

    def create(*args, **kwargs):
        created.append((args, kwargs))
        return fake

    monkeypatch.setattr(job.modal.Sandbox, "create", create)
    if times_out:
        with pytest.raises(api.Malformed) as exc:
            job._score_in_sandbox(object(), object(), "source", "a" * 12, b"input", 7, 150)
        assert exc.value.reason == "timeout"
    else:
        scores, runtime = job._score_in_sandbox(object(), object(), "source", "a" * 12, b"input", 7, 150)
        assert np.array_equal(scores, np.arange(7)) and runtime == 45.0
    assert created[0][1]["timeout"] == 300
    assert executed[0]["timeout"] == 270
    assert int(created[0][0][1]) > executed[0]["timeout"]
    assert terminated == [True]


@pytest.mark.parametrize("subset", [False, True])
def test_extended_container_time_does_not_relax_registered_score_limit(tmp_path, monkeypatch, subset):
    text = CANARIES[0]["text"]
    path = tmp_path / f"{api.proposal_sha(text)}.py"
    path.write_text(text)
    train, evaluation = _arrays(6), _arrays(7, first=10)
    train["labels"] = np.zeros(6, np.int64)
    input_path = tmp_path / "input.npz"
    input_path.write_bytes(pack_input(train, evaluation))
    output_path, status_path = tmp_path / "output.npz", tmp_path / "status.json"
    clock = iter([0.0, 61.0])
    with monkeypatch.context() as patch:
        patch.setattr(api.time, "perf_counter", lambda: next(clock))
        worker(str(path), str(input_path), str(output_path), str(status_path), subset=subset)
    assert json.loads(status_path.read_text()) == {"status": "timeout"}
    assert not output_path.exists()


def test_parent_deadlines_cover_nested_workers_within_modal_limit():
    from modal_jobs import seal

    assert job.SINGLE_SEED_JOB_TIMEOUT > 2 * job.SANDBOX_TIMEOUT
    assert job.CELL_JOB_TIMEOUT > 10 * job.SANDBOX_TIMEOUT
    assert seal.SEALED_CELL_TIMEOUT > job.CELL_JOB_TIMEOUT
    assert seal.SEALED_BATCH_TIMEOUT >= seal.MAX_BATCH_SIZE * job.CELL_JOB_TIMEOUT
    assert seal.SEALED_BATCH_TIMEOUT <= 24 * 60 * 60
