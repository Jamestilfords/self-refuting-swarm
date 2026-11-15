"""Registered baselines and in-scope ECE. One definition, shared with the experimenter.

``load_cache`` copies ``{split}__pooled`` from ``pooled.npz`` into
``run["splits"][split]["pooled"]`` when that file is present. ``logp_from_run``
raises ValueError naming the missing feature when ``pooled`` was not saved, so
the loader has to merge ``pooled.npz`` before ``d_full`` can be computed.
Nothing in this module reads a test split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from skeptic.critic.er1 import (
    dirichlet_ece,
    ece_from_probs,
    logit_scores,
    logits_to_alpha,
    logp_from_run,
    softmax_ece,
)
from skeptic.critic.types import BaselineSet, ScoreSet

G2_NAMES = (
    "energy_ce",
    "msp_ce",
    "d_port_ce",
    "d_full_ce",
    "energy_edl",
    "msp_edl",
    "d_port_edl",
    "d_full_edl",
)
REGISTERED_NAMES = G2_NAMES + ("vacuity_edl",)


def ece_in_scope(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE of a proposal's class probabilities on in-scope rows only.

    ``n_bins`` is accepted so the call site matches the registered bin count.
    ER-1's ``ece_from_probs`` uses the 15-bin metric.
    """
    if n_bins != 15:
        raise ValueError("ER-1 ECE uses 15 bins")
    probs = np.asarray(probs, dtype=np.float64)
    return float(ece_from_probs(probs.max(1), probs.argmax(1) == np.asarray(labels)))


def compute_baselines(run_ce: list[dict], run_edl: list[dict], cell: str) -> BaselineSet:
    if not run_ce or not run_edl:
        raise ValueError("both arms need at least one seed")
    if [int(run["meta"].get("seed", 0)) for run in run_ce] != [int(run["meta"].get("seed", 0)) for run in run_edl]:
        raise ValueError("CE and EDL caches are not aligned by seed")
    buckets: dict[str, dict[str, list]] = {
        name: {"id": [], "oos": []} for name in REGISTERED_NAMES
    }
    plain = {"ce": [], "edl": []}
    for run in run_ce:
        _logit_arm(run, "ce", buckets)
        _density_arm(run, "ce", buckets)
        val = run["splits"]["val"]
        plain["ce"].append(float(softmax_ece(val["logits"], val["labels"])))
    for run in run_edl:
        _logit_arm(run, "edl", buckets)
        _density_arm(run, "edl", buckets)
        val = run["splits"]["val"]
        alpha = logits_to_alpha(torch.from_numpy(np.asarray(val["logits"], dtype=np.float64)), 15.0, 1e-6)
        plain["edl"].append(float(dirichlet_ece(alpha.numpy(), val["labels"])))
        vacuity = logit_scores(val["logits"], arm="edl")["vacuity"]
        vacuity_oos = logit_scores(run["splits"]["oos_val"]["logits"], arm="edl")["vacuity"]
        buckets["vacuity_edl"]["id"].append(np.asarray(vacuity, dtype=np.float64))
        buckets["vacuity_edl"]["oos"].append(np.asarray(vacuity_oos, dtype=np.float64))

    scores = {}
    for name in REGISTERED_NAMES:
        arm = "edl" if name.endswith("_edl") else "ce"
        scores[name] = ScoreSet(
            name=name,
            arm=arm,
            val_id=tuple(buckets[name]["id"]),
            val_oos=tuple(buckets[name]["oos"]),
        )
    return BaselineSet(
        cell=cell,
        scores=scores,
        g2_names=G2_NAMES,
        plain_ece={arm: tuple(values) for arm, values in plain.items()},
    )


def _logit_arm(run: dict, arm: str, buckets: dict) -> None:
    for split, key in (("val", "id"), ("oos_val", "oos")):
        logits = run["splits"][split]["logits"]
        scored = logit_scores(logits, arm=arm)
        buckets[f"energy_{arm}"][key].append(np.asarray(scored["energy"], dtype=np.float64))
        buckets[f"msp_{arm}"][key].append(np.asarray(scored["msp"], dtype=np.float64))


def _density_arm(run: dict, arm: str, buckets: dict) -> None:
    port = logp_from_run(run, feature="feats", pca_dim=128)
    full = logp_from_run(run, feature="pooled", pca_dim=None)
    buckets[f"d_port_{arm}"]["id"].append(-np.asarray(port["val"], dtype=np.float64))
    buckets[f"d_port_{arm}"]["oos"].append(-np.asarray(port["oos_val"], dtype=np.float64))
    buckets[f"d_full_{arm}"]["id"].append(-np.asarray(full["val"], dtype=np.float64))
    buckets[f"d_full_{arm}"]["oos"].append(-np.asarray(full["oos_val"], dtype=np.float64))


def cell_cache_dir(cache_root: str | Path, cell: str) -> Path:
    benchmark, backbone = cell.split("/")
    return Path(cache_root) / f"{benchmark}_{backbone}"


def write_baselines(baseline: BaselineSet, out_dir: str | Path) -> None:
    out = Path(out_dir)
    for name, score in baseline.scores.items():
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        for index, (ident, oos) in enumerate(zip(score.val_id, score.val_oos)):
            np.save(folder / f"seed{index}.npy", np.concatenate([ident, oos]))
    payload = {arm: list(values) for arm, values in baseline.plain_ece.items()}
    (out / "plain_ece.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def main() -> None:
    import argparse

    from skeptic.critic.er1 import load_runs

    parser = argparse.ArgumentParser(description="Cache registered baseline scores for one cell.")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = cell_cache_dir(args.cache_root, args.cell)
    run_ce = load_runs(root, arm="ce")
    run_edl = load_runs(root, arm="edl")
    write_baselines(compute_baselines(run_ce, run_edl, args.cell), args.out)


if __name__ == "__main__":
    main()
