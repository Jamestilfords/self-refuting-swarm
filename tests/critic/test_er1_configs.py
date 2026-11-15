"""Three ER-1 configurations. Skipped unless SKEPTIC_CACHE_ROOT points at run caches."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from skeptic.critic import CriticConfig, LordPlusPlus, verdict
from skeptic.critic.baselines import cell_cache_dir, compute_baselines
from skeptic.critic.er1 import load_runs, spearman
from skeptic.critic.types import BaselineSet, ScoreSet

pytestmark = pytest.mark.er1cache


def _root() -> Path | None:
    text = os.environ.get("SKEPTIC_CACHE_ROOT")
    if not text:
        return None
    path = Path(text)
    return path if path.exists() else None


def _runs(cache_root: Path, cell: str, arm: str) -> list[dict]:
    return load_runs(cell_cache_dir(cache_root, cell), arm=arm)


def _density(runs: list[dict], *, whiten: bool, name: str) -> ScoreSet:
    from bert_daedl.config import GDAConfig
    from bert_daedl.posthoc.scoring import density_maps

    ident, oos = [], []
    config = GDAConfig(cov_mode="tied", pca_dim=128, whiten=whiten)
    for run in runs:
        logp = density_maps(run, feature="feats", gda_cfg=config)[1]
        ident.append(-np.asarray(logp["val"], dtype=np.float64))
        oos.append(-np.asarray(logp["oos_val"], dtype=np.float64))
    return ScoreSet(name=name, arm="edl", val_id=tuple(ident), val_oos=tuple(oos))


def _show(title: str, decision) -> None:
    ci = decision.ci
    ci_text = "n/a" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
    delta = "n/a" if decision.delta_val is None else f"{decision.delta_val:.4f}"
    p_value = "n/a" if decision.p is None else f"{decision.p:.4g}"
    alpha = "n/a" if decision.alpha_t is None else f"{decision.alpha_t:.5f}"
    print(f"{title:48} {decision.verdict.value:16} delta={delta:8} ci={ci_text:24} p={p_value:8} alpha={alpha}")


def test_three_known_configurations(capsys):
    cache_root = _root()
    if cache_root is None:
        pytest.skip("skipped: no caches")
    config = CriticConfig(bootstrap_B=2000)
    with capsys.disabled():
        print("\nconfig                                            verdict          delta    ci                       p        alpha")

        edl = _runs(cache_root, "clinc150/bert", "edl")
        whitened = _density(edl, whiten=True, name="d_port_whiten")
        raw = _density(edl, whiten=False, name="d_port_raw")
        shell = BaselineSet(
            cell="clinc150/bert",
            scores={raw.name: raw, whitened.name: whitened},
            g2_names=(),
            plain_ece={"edl": tuple(0.0 for _ in edl), "ce": tuple(0.0 for _ in edl)},
        )
        first = verdict(whitened, raw, shell, LordPlusPlus(), "clinc150/bert", config=config)
        _show("1 whitening after PCA", first)
        assert first.verdict.value == "INERT"

        ce = _runs(cache_root, "clinc150/bert", "ce")
        baselines = compute_baselines(ce, edl, "clinc150/bert")
        rhos = []
        vacuity = baselines.scores["vacuity_edl"]
        energy = baselines.scores["energy_edl"]
        for index in range(len(vacuity.val_id)):
            rhos.append(spearman(
                np.concatenate([vacuity.val_id[index], vacuity.val_oos[index]]),
                np.concatenate([energy.val_id[index], energy.val_oos[index]]),
            ))
        print(f"  spearman(vacuity_edl, energy_edl) per seed: {['%.4f' % rho for rho in rhos]}")
        second = verdict(
            vacuity,
            baselines.scores["msp_ce"],
            baselines,
            LordPlusPlus(),
            "clinc150/bert",
            config=config,
        )
        reason = second.detail.get("G2", {})
        print(f"  G2 matched_by={reason.get('matched_by')} gain_inc={reason.get('gain_inc')} gain_base={reason.get('gain_base')}")
        _show("2 vacuity_edl vs msp_ce", second)
        assert second.verdict.value == "SCORE-MATCHED"

        ce_b = _runs(cache_root, "banking77_oos/bert", "ce")
        edl_b = _runs(cache_root, "banking77_oos/bert", "edl")
        bank = compute_baselines(ce_b, edl_b, "banking77_oos/bert")
        third = verdict(
            bank.scores["d_port_edl"],
            bank.scores["vacuity_edl"],
            bank,
            LordPlusPlus(),
            "banking77_oos/bert",
            config=config,
            exclude_baselines=frozenset({"d_port_edl"}),
        )
        _show("3 d_port_edl vs vacuity_edl", third)
        if third.verdict.value == "UNDERPOWERED":
            print(
                f"  validation-power: delta={third.delta_val:.4f} "
                f"se={third.detail['G4']['se_combined']:.4f} alpha_1=0.00268"
            )
        assert third.verdict.value in {"ACCEPT", "UNDERPOWERED"}
