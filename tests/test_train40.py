"""E2 plan and cache boundary checks; these do not start Modal jobs."""

import json

import numpy as np
import pytest

from modal_jobs import train40


def test_plan_has_forty_unique_runs():
    runs = train40.plan_runs()
    assert len(runs) == 40 == len({run.name for run in runs})
    assert runs[0].name == "clinc150_bert/ce_seed0"
    assert runs[-1].name == "banking77_roberta/edl_seed4"
    assert train40.TRAIN40_MODAL_OPTIONS["gpu"] == "B300"
    assert train40.TRAIN40_MODAL_OPTIONS["max_containers"] == 4


def test_open_cache_omits_test_and_validation_labels(tmp_path):
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    arrays, pooled = {}, {}
    for split in ("train", "val", "oos_val", "test", "oos_test"):
        arrays[f"{split}__logits"] = np.ones((2, 3), dtype=np.float32)
        arrays[f"{split}__feats"] = np.ones((2, 4), dtype=np.float32)
        arrays[f"{split}__labels"] = np.array([0, 1])
        pooled[f"{split}__pooled"] = np.ones((2, 5), dtype=np.float32)
    np.savez(sealed / "cache.npz", **arrays)
    np.savez(sealed / "pooled.npz", **pooled)

    recipe = train40._recipe(train40.plan_runs()[0])
    opened = tmp_path / "open" / "clinc150_bert" / "ce_seed0"
    train40.publish_open_cache(sealed, opened, recipe)
    with np.load(opened / "cache.npz", allow_pickle=False) as cache:
        assert set(cache.files) == {
            f"{split}__{field}" for split in ("train", "val", "oos_val")
            for field in ("logits", "feats", "pooled")
        } | {"train__labels"}
    assert not any("test" in p.name for p in opened.iterdir())
    assert json.loads((opened / "manifest.json").read_text())["recipe_hash"] == train40._fingerprint(recipe)

    train40.publish_open_cache(sealed, opened, recipe)  # identical retry is safe
    with pytest.raises(RuntimeError, match="different or incomplete"):
        train40.publish_open_cache(sealed, opened, {**recipe, "seed": 99})

    # A matching manifest cannot make an expanded open cache acceptable.
    with np.load(opened / "cache.npz", allow_pickle=False) as cache:
        expanded = {name: cache[name] for name in cache.files}
    expanded["test__labels"] = np.array([0, 1])
    np.savez(opened / "cache.npz", **expanded)
    with pytest.raises(RuntimeError, match="different or incomplete"):
        train40.publish_open_cache(sealed, opened, recipe)


def test_open_cache_rejects_nonfinite_values(tmp_path):
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    arrays = {f"{split}__{field}": np.array([float("nan") if split == "val" else 1.0])
              for split in ("train", "val", "oos_val") for field in ("logits", "feats", "labels")}
    pooled = {f"{split}__pooled": np.array([1.0]) for split in ("train", "val", "oos_val")}
    np.savez(sealed / "cache.npz", **arrays)
    np.savez(sealed / "pooled.npz", **pooled)
    opened = tmp_path / "open" / "run"
    with pytest.raises(RuntimeError, match="Non-finite"):
        train40.publish_open_cache(sealed, opened, {"seed": 0})
    assert not opened.exists()
