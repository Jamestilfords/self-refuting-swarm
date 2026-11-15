"""Isolated scoring tests; Modal calls are mocked, no cloud jobs run."""
import json
from pathlib import Path
import numpy as np
import pytest
from modal_jobs import score_stream
from modal_jobs.train40 import RunSpec, _fingerprint, _recipe
from skeptic import api
from skeptic.score_artifacts import load_scores
from skeptic.sandbox import unpack_input
from skeptic.streams.canaries import CANARIES

def _open_caches(root: Path):
    for seed in range(5):
        spec = RunSpec("clinc150", "bert", "ce", seed)
        directory = root / spec.name
        directory.mkdir(parents=True)
        arrays = {}
        for split, start in (("train", 10), ("val", 1), ("oos_val", 5)):
            logits = np.zeros((4, 150), np.float32)
            logits[:, 0] = np.arange(start, start + 4)
            arrays[f"{split}__logits"] = logits
            arrays[f"{split}__feats"] = np.ones((4, 256), np.float32)
            arrays[f"{split}__pooled"] = np.ones((4, 768), np.float32)
        arrays["train__labels"] = np.zeros(4, np.int64)
        np.savez_compressed(directory / "cache.npz", **arrays)
        recipe = _recipe(spec)
        (directory / "manifest.json").write_text(json.dumps(
            {"run": recipe, "recipe_hash": _fingerprint(recipe)}), encoding="utf-8")

@pytest.mark.parametrize("with_probs", [False, True])
def test_isolated_validation_scorer_writes_aligned_per_item_artifact(tmp_path, monkeypatch, with_probs):
    _open_caches(tmp_path)
    monkeypatch.setattr(score_stream, "OPEN_MOUNT", str(tmp_path))
    monkeypatch.setattr(score_stream.modal.App, "lookup", staticmethod(lambda *args, **kw: object()))
    monkeypatch.setattr(score_stream.modal.Image, "from_id", staticmethod(lambda *args, **kw: object()))
    monkeypatch.setattr(api, "load_proposal", lambda *args: pytest.fail("proposal ran in trusted process"))

    class Volume:
        def reload(self):
            pass

        def commit(self):
            pass

    monkeypatch.setattr(score_stream, "open_volume", Volume())
    calls = []

    def score(_owner, _image, _text, _sha, payload, n, k, *, subset=False, return_probs=False):
        ctx = unpack_input(payload)
        assert ctx.eval.labels is None and len(ctx.eval) == n and k == 150
        calls.append(subset)
        values = np.asarray(ctx.eval.logits[:, 0], np.float64)
        probs = np.zeros((n, k)); probs[:, 0] = 1.0
        return (values, probs if with_probs else None, .01) if return_probs else (values, .01)

    monkeypatch.setattr(score_stream, "_score_in_sandbox", score)
    text = CANARIES[0]["text"]
    sha = api.proposal_sha(text)
    task = {"cell": "clinc150/bert", "sha": sha, "text": text, "arm": "ce"}
    with pytest.warns(UserWarning, match="executing locally"):
        result = score_stream.score_one.local(task, "im-test")
    row = result["rows"][0]
    assert row["status"] == "ok" and row["val"]["auroc"] == [1.] * 5
    assert calls == [False, True] * 5
    path = tmp_path / "scores_isolated/clinc150_bert" / f"{sha}.npz"
    with np.load(path, allow_pickle=False) as saved:
        assert np.array_equal(saved["scores"][0], np.arange(1, 9))
        assert str(saved["scorer"]) == "isolated-e3-v1"
    artifact = load_scores(tmp_path, "clinc150/bert", sha, source="isolated")
    assert artifact.scores.shape == (5, 8) and artifact.n_in == 4
    assert (artifact.pred is not None) == with_probs
    with pytest.warns(UserWarning, match="executing locally"):
        again = score_stream.score_one.local(task, "im-test")
    assert again["rows"][0]["val"] == row["val"] and len(calls) == 10


def test_cli_preserves_calibration_in_ledger(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    text = CANARIES[0]["text"]
    sha = api.proposal_sha(text)
    proposals = tmp_path / "proposals"
    proposals.mkdir()
    (proposals / f"{sha}.py").write_text(text)
    (proposals / "baselines.json").write_text(json.dumps([{"sha": sha}]))
    monkeypatch.setattr(score_stream, "PROPOSALS", proposals)
    monkeypatch.setattr(score_stream, "REPO", tmp_path)
    monkeypatch.setattr(score_stream, "stream_app", SimpleNamespace(run=lambda **kw: nullcontext()))
    monkeypatch.setattr(score_stream.modal, "enable_output", nullcontext)
    monkeypatch.setattr(score_stream, "sandbox_image", SimpleNamespace(object_id="im-test"))
    def collect(jobs, image):
        return {(job["sha"], job["cell"]): {"sha": sha, "arm": "ce", "status": "ok",
                "val": {"auroc": [.7] * 5, "fpr95": [.4] * 5}, "has_probs": True}
                for job in jobs}
    def calibrate(tasks):
        return [{"cell": t["cell"], "rows": [{"sha": s, "val_ece": [.03] * 5,
                   "val_ece_plain": [.04] * 5} for s, arm in t["items"]]} for t in tasks]
    monkeypatch.setattr(score_stream, "_collect", collect)
    monkeypatch.setattr(score_stream, "calibrate_cell", SimpleNamespace(map=calibrate))
    assert score_stream.main(["run", "--streams", "baseline"]) == 0
    rows = [json.loads(line) for line in (tmp_path / "runs/ledger_isolated_partial.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert all(r["val"]["ece"] == [.03] * 5 and r["sealed"] == "not_run" for r in rows)
    assert not (tmp_path / "runs/ledger.jsonl").exists()
    monkeypatch.setattr(score_stream, "calibrate_cell", SimpleNamespace(map=lambda tasks: []))
    with pytest.raises(RuntimeError, match="incomplete"):
        score_stream.main(["run", "--streams", "baseline"])


def test_legacy_seal_flag_is_unavailable():
    with pytest.raises(SystemExit) as exc:
        score_stream.main(["run", "--seal"])
    assert exc.value.code == 2


def test_artifact_provenance_and_shape_are_required(tmp_path):
    sha = "a" * 12
    path = tmp_path / "scores_isolated/clinc150_bert" / f"{sha}.npz"
    path.parent.mkdir(parents=True)
    np.savez_compressed(path, scores=np.zeros((5, 8)), n_in=4)
    with pytest.raises(ValueError, match="provenance"):
        load_scores(tmp_path, "clinc150/bert", sha, source="isolated")
    np.savez_compressed(path, scores=np.zeros((4, 8)), n_in=4,
                        scorer="isolated-e3-v1", proposal_sha=sha)
    with pytest.raises(ValueError, match="five finite"):
        load_scores(tmp_path, "clinc150/bert", sha, source="isolated")
