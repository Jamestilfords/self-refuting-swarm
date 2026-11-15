"""Registered scores from a tiny fake run, plus the whitening invariance."""

from __future__ import annotations

import numpy as np
from scipy.special import softmax

from bert_daedl.config import GDAConfig
from bert_daedl.posthoc.scoring import density_maps
from bert_daedl.posthoc.gate import pair_decomposition
from skeptic.critic.baselines import REGISTERED_NAMES, compute_baselines


def _run(arm: str, seed: int, rng: np.random.Generator) -> dict:
    def split(n_rows: int, n_classes: int | None):
        logits = rng.normal(size=(n_rows, 5))
        feats = rng.normal(size=(n_rows, 16))
        pooled = rng.normal(size=(n_rows, 12))
        if n_classes is None:
            labels = np.full(n_rows, -1)
        else:
            labels = np.arange(n_rows) % n_classes
        return {"logits": logits, "feats": feats, "pooled": pooled, "labels": labels}

    return {
        "path": f"fake/{arm}/{seed}",
        "meta": {"arm": arm, "seed": seed},
        "splits": {
            "train": split(80, 5),
            "val": split(40, 5),
            "oos_val": split(20, None),
        },
    }


def test_compute_baselines_shapes_and_msp():
    rng = np.random.default_rng(0)
    run_ce = [_run("ce", 0, rng), _run("ce", 1, rng)]
    run_edl = [_run("edl", 0, rng), _run("edl", 1, rng)]
    baselines = compute_baselines(run_ce, run_edl, "clinc150/bert")
    assert set(baselines.scores) == set(REGISTERED_NAMES)
    assert len(baselines.g2_names) == 8
    assert "vacuity_edl" not in baselines.g2_names
    for name, score in baselines.scores.items():
        assert len(score.val_id) == 2
        assert score.val_id[0].shape == (40,)
        assert score.val_oos[0].shape == (20,)
        assert np.isfinite(score.val_id[0]).all()
        assert np.isfinite(score.val_oos[0]).all()
    logits = run_ce[0]["splits"]["val"]["logits"]
    expected = 1.0 - softmax(logits, axis=-1).max(-1)
    assert np.allclose(baselines.scores["msp_ce"].val_id[0], expected, rtol=1e-6, atol=1e-6)


def test_whitening_does_not_change_the_ranking():
    rng = np.random.default_rng(1)
    run = _run("edl", 0, rng)
    whitened = density_maps(run, feature="feats", gda_cfg=GDAConfig(cov_mode="tied", pca_dim=8, whiten=True))[1]
    plain = density_maps(run, feature="feats", gda_cfg=GDAConfig(cov_mode="tied", pca_dim=8, whiten=False))[1]
    parts = pair_decomposition(-plain["val"], -plain["oos_val"], -whitened["val"], -whitened["oos_val"])
    assert parts["flip_rate"] < 0.001
