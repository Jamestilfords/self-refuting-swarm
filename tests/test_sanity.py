"""Validation scoring and open-volume boundary checks without Modal calls."""

import json

import numpy as np
import pytest

from modal_jobs import sanity, train40
from skeptic.api import Helpers


def test_auroc_handles_ties_and_direction():
    assert sanity.auroc(np.array([0, 0]), np.array([1, 1])) == 1.0
    assert sanity.auroc(np.array([1, 1]), np.array([0, 0])) == 0.0
    assert sanity.auroc(np.array([1, 1]), np.array([1, 1])) == 0.5


def test_logit_baselines_match_proposal_helpers():
    logits = np.random.default_rng(7).normal(size=(20, 5))
    scores = sanity.logit_scores(logits)
    assert np.allclose(scores["msp"], Helpers.msp(logits))
    assert np.allclose(scores["energy"], Helpers.energy(logits))
    assert np.allclose(scores["vacuity"], Helpers.vacuity(logits))


def _write_open_cache(root, spec):
    directory = root / spec.name
    directory.mkdir(parents=True)
    n_classes = 150 if spec.dataset == "clinc150" else 50
    val = np.zeros((4, n_classes), dtype=np.float32)
    val[:, 0] = 8
    oos = np.zeros((3, n_classes), dtype=np.float32)
    arrays = {}
    for name, logits in (("train", val), ("val", val), ("oos_val", oos)):
        arrays[f"{name}__logits"] = logits
        arrays[f"{name}__feats"] = np.zeros((len(logits), 256), dtype=np.float32)
        arrays[f"{name}__pooled"] = np.zeros((len(logits), 768), dtype=np.float32)
    arrays["train__labels"] = np.zeros(len(val), dtype=np.int64)
    np.savez_compressed(directory / "cache.npz", **arrays)
    recipe = train40._recipe(spec)
    (directory / "manifest.json").write_text(json.dumps({"run": recipe,
        "recipe_hash": train40._fingerprint(recipe)}))
    return directory


def test_audit_all_requires_forty_valid_caches(tmp_path):
    specs = train40.plan_runs()
    for spec in specs:
        _write_open_cache(tmp_path, spec)
    report = sanity.audit_all(tmp_path)
    assert report["status"] == "ok" and report["n_runs"] == 40
    assert len(report["groups"]) == 8
    assert all(row["primary_auroc"] == 1.0 for row in report["rows"])
    assert all(group["mean_validation_auroc"] == 1.0 for group in report["groups"])

    (tmp_path / specs[-1].name / "manifest.json").unlink()
    with pytest.raises(RuntimeError, match="Missing open cache"):
        sanity.audit_all(tmp_path)


def test_audit_rejects_test_arrays_in_open_cache(tmp_path):
    spec = train40.plan_runs()[0]
    directory = _write_open_cache(tmp_path, spec)
    with np.load(directory / "cache.npz", allow_pickle=False) as cache:
        arrays = {name: cache[name] for name in cache.files}
    arrays["test__labels"] = np.array([0])
    np.savez_compressed(directory / "cache.npz", **arrays)
    with pytest.raises(RuntimeError, match="Unexpected or missing arrays"):
        sanity.audit_run(tmp_path, spec)
