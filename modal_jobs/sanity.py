"""E2 validation audit of the 40 open caches; never reads sealed test data.

Run locally with ``python -m modal_jobs.sanity`` while using the Modal profile
for the workspace where training ran. The CPU function mounts only the open
volume, checks every expected cache, and returns validation AUROC by run.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal
import numpy as np

from modal_jobs.app import open_volume, settings
from modal_jobs.config import OPEN_MOUNT
from modal_jobs.train40 import RunSpec, _fingerprint, _recipe, plan_runs


def auroc(id_scores: np.ndarray, oos_scores: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties; higher means more OOS."""
    inside = np.asarray(id_scores, dtype=np.float64).reshape(-1)
    outside = np.asarray(oos_scores, dtype=np.float64).reshape(-1)
    if not len(inside) or not len(outside):
        raise ValueError("Validation and OOS validation must both be nonempty")
    scores = np.concatenate((inside, outside))
    if not np.isfinite(scores).all():
        raise ValueError("Validation scores contain non-finite values")
    ordered = np.sort(scores)
    left = np.searchsorted(ordered, outside, side="left")
    right = np.searchsorted(ordered, outside, side="right")
    rank_sum = ((left + right + 1) / 2).sum()
    return float((rank_sum - len(outside) * (len(outside) + 1) / 2)
                 / (len(inside) * len(outside)))


def logit_scores(logits: np.ndarray) -> dict[str, np.ndarray]:
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or not len(z) or z.shape[1] < 2 or not np.isfinite(z).all():
        raise ValueError("Expected finite, nonempty two-dimensional logits")
    shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    probabilities = exp_z / exp_z.sum(axis=1, keepdims=True)
    energy = -(z.max(axis=1) + np.log(exp_z.sum(axis=1)))
    alpha = np.exp(np.clip(z, -15, 15)) + 1e-6
    return {"msp": 1 - probabilities.max(axis=1),
            "energy": energy,
            "vacuity": z.shape[1] / alpha.sum(axis=1)}


def _allowed_keys() -> set[str]:
    keys = {f"{split}__{field}" for split in ("train", "val", "oos_val")
            for field in ("logits", "feats", "pooled")}
    return keys | {"train__labels"}


def audit_run(root: Path, spec: RunSpec) -> dict:
    directory = root / spec.name
    manifest_path, cache_path = directory / "manifest.json", directory / "cache.npz"
    if not manifest_path.is_file() or not cache_path.is_file():
        raise RuntimeError(f"Missing open cache for {spec.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recipe = _recipe(spec)
    if manifest.get("run") != recipe or manifest.get("recipe_hash") != _fingerprint(recipe):
        raise RuntimeError(f"Recipe mismatch for {spec.name}")

    with np.load(cache_path, allow_pickle=False) as cache:
        if set(cache.files) != _allowed_keys():
            raise RuntimeError(f"Unexpected or missing arrays in open cache for {spec.name}")
        val = np.asarray(cache["val__logits"])
        oos = np.asarray(cache["oos_val__logits"])
    expected_classes = 150 if spec.dataset == "clinc150" else 50
    if val.ndim != 2 or oos.ndim != 2 or val.shape[1] != expected_classes or oos.shape[1] != expected_classes:
        raise RuntimeError(f"Wrong logit shape for {spec.name}")
    val_scores, oos_scores = logit_scores(val), logit_scores(oos)
    metrics = {name: auroc(val_scores[name], oos_scores[name]) for name in val_scores}
    primary = "msp" if spec.arm == "ce" else "vacuity"
    return {"run": spec.name, "dataset": spec.dataset, "backbone": spec.backbone,
            "arm": spec.arm, "seed": spec.seed, "n_val": len(val), "n_oos_val": len(oos),
            "auroc": metrics, "primary_score": primary, "primary_auroc": metrics[primary]}


def audit_all(root: Path) -> dict:
    rows = [audit_run(root, spec) for spec in plan_runs()]
    warnings = [f"{row['run']}: primary validation AUROC is {row['primary_auroc']:.3f} (at or below chance)"
                for row in rows if row["primary_auroc"] <= 0.5]
    groups = []
    for dataset in ("clinc150", "banking77_oos"):
        for backbone in ("bert", "roberta"):
            for arm in ("ce", "edl"):
                group = [row for row in rows if (row["dataset"], row["backbone"], row["arm"])
                         == (dataset, backbone, arm)]
                if len(group) != 5 or {row["seed"] for row in group} != set(range(5)):
                    raise RuntimeError(f"Missing seed in {dataset}/{backbone}/{arm}")
                groups.append({"dataset": dataset, "backbone": backbone, "arm": arm,
                               "primary_score": group[0]["primary_score"],
                               "mean_validation_auroc": float(np.mean([row["primary_auroc"] for row in group]))})
    return {"status": "needs_review" if warnings else "ok", "n_runs": len(rows),
            "warnings": warnings, "rows": rows, "groups": groups}


sanity_app = modal.App(f"{settings.app_name}-sanity")
sanity_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==1.26.4")
    .env(settings.container_env())
    .add_local_python_source("modal_jobs")
)


@sanity_app.function(image=sanity_image,
                     volumes={OPEN_MOUNT: open_volume.with_mount_options(read_only=True)},
                     cpu=2, memory=8192, timeout=600)
def score_validation() -> dict:
    report = audit_all(Path(OPEN_MOUNT))
    print(json.dumps({"status": report["status"], "n_runs": report["n_runs"],
                      "warnings": report["warnings"], "groups": report["groups"]}, sort_keys=True))
    return report


def main() -> None:
    with modal.enable_output():
        with sanity_app.run(environment_name=settings.environment):
            report = score_validation.remote()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
