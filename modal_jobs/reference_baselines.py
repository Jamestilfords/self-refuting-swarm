"""Step 2 of the unblinding plan: write the registered baseline arrays for check_references.

    python -m modal_jobs.reference_baselines
    modal volume get skeptic-open baselines runs/ --force
    python -m skeptic.analysis.check_references

The registered baselines need validation labels (for the plain-model ECE), which only the sealed volume
holds. This job mounts the sealed volume read-only and reads only the train, val and oos_val arrays of each
cache, plus cache_meta.json. It never accesses a test or oos_test array; the keys it read are returned and
printed. It writes runs/baselines/<group>/<name>/seed<k>.npy and plain_ece.json to the open volume and
returns no scores.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

from modal_jobs.app import open_volume, sealed_volume, settings
from modal_jobs.config import OPEN_MOUNT, SEALED_MOUNT

REPO = Path(__file__).resolve().parents[1]
ER1_SOURCE = REPO / "third_party" / "er1" / "src"
SPLITS = ("train", "val", "oos_val")
ref_app = modal.App(f"{settings.app_name}-reference-baselines")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.4", "scipy==1.18.1", "scikit-learn==1.9.1")
    .pip_install("torch==2.15.0", index_url="https://download.pytorch.org/whl/cpu")
    .env({**settings.container_env(), "SKEPTIC_ER1_SRC": "/opt/er1/src", "PYTHONPATH": "/opt/er1/src"})
)
if ER1_SOURCE.is_dir():
    image = image.add_local_dir(ER1_SOURCE, "/opt/er1/src", copy=True)
image = image.add_local_python_source("skeptic", "modal_jobs")


def _load_run(run_dir: Path, read: list) -> dict:
    """ER-1 load_cache, restricted to the validation-side splits. npz members are read only when accessed."""
    meta = json.loads((run_dir / "cache_meta.json").read_text(encoding="utf-8"))
    splits = {}
    with open(run_dir / "cache.npz", "rb") as f1, open(run_dir / "pooled.npz", "rb") as f2:
        import numpy as np
        blob, pooled = np.load(f1), np.load(f2)
        for split in SPLITS:
            keys = [f"{split}__logits", f"{split}__feats", f"{split}__labels"]
            splits[split] = {k.split("__")[1]: np.asarray(blob[k]) for k in keys}
            splits[split]["pooled"] = np.asarray(pooled[f"{split}__pooled"])
            read += [f"{run_dir.name}/cache.npz:{k}" for k in keys] + [f"{run_dir.name}/pooled.npz:{split}__pooled"]
    if any(not k.split(":")[1].split("__")[0] in SPLITS for k in read):
        raise RuntimeError("a non-validation array was read")
    return {"path": run_dir, "meta": meta, "splits": splits}


@ref_app.function(image=image, cpu=4, memory=16 * 1024, timeout=60 * 60,
                  volumes={SEALED_MOUNT: sealed_volume.with_mount_options(read_only=True), OPEN_MOUNT: open_volume})
def write_cell(cell: str, group: str) -> dict:
    from skeptic.critic.baselines import compute_baselines, write_baselines
    read: list = []
    runs = {arm: [_load_run(Path(SEALED_MOUNT) / group / f"{arm}_seed{k}", read) for k in range(5)] for arm in ("ce", "edl")}
    out = Path(OPEN_MOUNT) / "baselines" / group
    write_baselines(compute_baselines(runs["ce"], runs["edl"], cell), out)
    open_volume.commit()
    return {"cell": cell, "written": sorted(p.parent.name for p in out.glob("*/seed0.npy")),
            "arrays_read": len(read), "splits_read": sorted({k.split(":")[1].split("__")[0] for k in read})}


def main() -> int:
    from skeptic.score_artifacts import CELLS
    with modal.enable_output():
        with ref_app.run(environment_name=settings.environment):
            for result in write_cell.starmap(list(CELLS.items())):
                print(json.dumps(result))
    print("next: modal volume get skeptic-open baselines runs/ --force")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
