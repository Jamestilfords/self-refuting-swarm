"""Guarded Modal sealer on synthetic caches; network and compute are mocked."""
import io
import json
from pathlib import Path
import numpy as np
import pytest
from modal_jobs import seal
from modal_jobs.train40 import RunSpec
from skeptic import api, store
from skeptic.sandbox import unpack_input
from skeptic.streams.canaries import CANARIES
from tests.test_store import decisions, HEAD, FIRST, CELLS

def _sealed_caches(root: Path):
    for seed in range(5):
        spec = RunSpec("clinc150", "bert", "ce", seed)
        directory = root / spec.name
        directory.mkdir(parents=True)
        arrays = {}
        pooled = {}
        for split, start in (("train", 10), ("test", 1), ("oos_test", 5)):
            n = 4
            logits = np.zeros((n, 150), np.float32)
            logits[:, 0] = np.arange(start, start + n)
            arrays[f"{split}__logits"] = logits
            arrays[f"{split}__feats"] = np.ones((n, 256), np.float32)
            pooled[f"{split}__pooled"] = np.ones((n, 768), np.float32)
            arrays[f"{split}__labels"] = np.zeros(n, np.int64)
        np.savez_compressed(directory / "cache.npz", **arrays)
        np.savez_compressed(directory / "pooled.npz", **pooled)

def test_sealed_guard_and_anonymous_test_payload(tmp_path, monkeypatch):
    _sealed_caches(tmp_path)
    sealed = store.SealedStore(tmp_path, tmp_path)
    spec = RunSpec("clinc150", "bert", "ce", 0)
    assert len(sealed.read_array(spec.name, "train", "labels")) == 4
    with pytest.raises(store.GuardError, match="sealed"):
        sealed.read_array(spec.name, "test", "logits")
    monkeypatch.setattr(store, "verify_remote_freeze", lambda c, d: store.UnsealPermit(d, c))
    sealed.unseal_remote("a" * 40, "b" * 64)
    full, subset, permutation, subset_idx, n_in, k = seal.prepare_test_cell(sealed, spec)
    ctx = unpack_input(full)
    assert n_in == 4 and k == 150 and len(permutation) == 8
    assert ctx.eval.labels is None and len(unpack_input(subset).eval) == len(subset_idx) == 2
    assert b"test__labels" not in full and b"oos_test__labels" not in full
    with pytest.raises(store.GuardError, match="Unknown E2 run"):
        sealed.read_array("../../other", "test", "logits")

def test_isolated_sealer_returns_status_and_keeps_scores_sealed(tmp_path, monkeypatch):
    _sealed_caches(tmp_path)
    monkeypatch.setattr(seal, "SEALED_MOUNT", str(tmp_path))
    monkeypatch.setattr(store, "verify_remote_freeze", lambda c, d: store.UnsealPermit(d, c))
    monkeypatch.setattr(seal, "verify_remote_proposal", lambda c, cell, sha: None)
    monkeypatch.setattr(seal.modal.App, "lookup", staticmethod(lambda *args, **kw: object()))
    monkeypatch.setattr(seal.modal.Image, "from_id", staticmethod(lambda *args, **kw: object()))

    commits = []

    class Volume:
        def reload(self):
            pass

        def commit(self):
            assert not list(tmp_path.rglob("*.tmp.npz"))
            commits.append(True)

    monkeypatch.setattr(seal, "sealed_volume", Volume())
    calls = []

    def score(_owner, _image, _text, _sha, payload, n, k, *, subset=False):
        ctx = unpack_input(payload)
        assert ctx.eval.labels is None and len(ctx.eval) == n and k == 150
        calls.append(subset)
        return np.asarray(ctx.eval.logits[:, 0], np.float64), 0.01

    monkeypatch.setattr(seal, "_score_in_sandbox", score)
    text = CANARIES[0]["text"]
    with pytest.warns(UserWarning, match="executing locally"):
        result = seal.score_sealed.local(text, "clinc150", "bert", "ce", "a" * 40, "b" * 64, "im-test")
    assert result == {"sha": api.proposal_sha(text), "cell": "clinc150/bert", "status": "sealed"}
    assert calls == [False, True] * 5 and commits == [True]
    path = tmp_path / "scores/clinc150_bert" / f"{api.proposal_sha(text)}.npz"
    with np.load(path, allow_pickle=False) as saved:
        assert saved["scores"].shape == (5, 8) and int(saved["n_in"]) == 4
        assert np.array_equal(saved["scores"][0], np.arange(1, 9))
    with pytest.warns(UserWarning, match="executing locally"):
        again = seal.score_sealed.local(text, "clinc150", "bert", "ce", "a" * 40, "b" * 64, "im-test")
    assert again["status"] == "already_sealed" and len(calls) == 10


def _publication(repo, *, head=HEAD):
    def download(url, limit):
        if url.endswith("/git/ref/heads/main"):
            return json.dumps({"object": {"sha": head}}).encode()
        prefix = f"https://raw.githubusercontent.com/{store.PUBLISHED_REPO}/{HEAD}/"
        assert url.startswith(prefix)
        return (repo / url[len(prefix):]).read_bytes()
    return download


def test_remote_guard_recomputes_complete_freeze_and_proposal_membership(decisions, monkeypatch):
    digest = store.write_decision_hash(decisions)
    monkeypatch.setattr(store, "_download", _publication(decisions))
    assert store.verify_remote_freeze(HEAD, digest) == store.UnsealPermit(digest, HEAD)
    store.verify_remote_proposal(HEAD, CELLS[0], FIRST)
    with pytest.raises(store.GuardError, match="not in the published"):
        store.verify_remote_proposal(HEAD, CELLS[0], "f" * 12)
    path = decisions / "runs/decisions/R1/ord_000.jsonl"
    path.write_text(path.read_text().replace('"idx": 0', '"idx":  0'))
    with pytest.raises(store.GuardError, match="differs from the logs"):
        store.verify_remote_freeze(HEAD, digest)


@pytest.mark.parametrize("damage", ["main", "hash", "registration", "network"])
def test_remote_guard_fails_closed_and_revokes_permit(decisions, monkeypatch, damage):
    digest = store.write_decision_hash(decisions)
    monkeypatch.setattr(store, "_download", _publication(decisions))
    sealed = store.SealedStore(decisions, decisions)
    sealed.unseal_remote(HEAD, digest)
    sealed.guard_read("test")
    if damage == "main":
        monkeypatch.setattr(store, "_download", _publication(decisions, head="e" * 40))
    elif damage == "hash":
        (decisions / store.HASH_RELATIVE).write_text("0" * 64 + "\n")
    elif damage == "registration":
        (decisions / "PREREGISTRATION.md").write_text("Status: DRAFT\n")
    else:
        def failed(url, limit):
            raise store.GuardError("network unavailable")
        monkeypatch.setattr(store, "_download", failed)
    with pytest.raises(store.GuardError):
        sealed.unseal_remote(HEAD, digest)
    with pytest.raises(store.GuardError, match="sealed"):
        sealed.guard_read("test")


def test_remote_download_is_bounded(monkeypatch):
    monkeypatch.setattr(store.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"x" * 33))
    with pytest.raises(store.GuardError, match="download limit"):
        store._download("https://example.invalid", 32)


def test_sealer_denies_unfrozen_proposal_before_test_reads(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(seal, "sealed_volume", SimpleNamespace(reload=lambda: None))
    monkeypatch.setattr(store, "verify_remote_freeze", lambda c, d: store.UnsealPermit(d, c))
    def reject(*args):
        raise store.GuardError("Proposal is not in the published decision set")
    monkeypatch.setattr(seal, "verify_remote_proposal", reject)
    monkeypatch.setattr(store.SealedStore, "read_array", lambda *a: pytest.fail("test read before authorization"))
    with pytest.warns(UserWarning, match="executing locally"), pytest.raises(store.GuardError, match="not in"):
        seal.score_sealed.local(CANARIES[0]["text"], "clinc150", "bert", "ce", HEAD, "a" * 64, "im-test")
