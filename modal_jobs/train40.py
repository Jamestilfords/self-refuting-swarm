"""E2: launch 40 independent ER-1 fine-tunes on Modal B300s.

Run ``python -m modal_jobs.train40 plan`` locally to preview the matrix, then
``python -m modal_jobs.train40 run --dataset clinc150 --backbone bert --arm ce
--seed 0`` for one run or ``python -m modal_jobs.train40 run --all`` for all 40.
Raw data and complete caches stay sealed; only train labels and label-free
validation features are copied to the open volume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import modal

from modal_jobs.app import app, open_volume, sealed_volume, settings
from modal_jobs.config import OPEN_MOUNT, SEALED_MOUNT

TRAIN40_MODAL_OPTIONS = {
    "gpu": "B300",
    "cpu": 8,
    "memory": 32 * 1024,  # MiB per fine-tune
    "max_containers": 4,
    "timeout": 2 * 60 * 60,  # seconds per fine-tune
}
DATASETS = ("clinc150", "banking77_oos")
BACKBONES = {
    "bert": ("google-bert/bert-base-uncased", "86b5e0934494bd15c9632b12f734a8a67f723594"),
    "roberta": ("FacebookAI/roberta-base", "e2da8e2f811d1448a5b465c236feacd80ffbac7b"),
}
ARMS = ("ce", "edl")
SEEDS = range(5)
ER1_COMMIT = "e877e9159d6c099ffdc06ac54d32aafc977aa437"
CLINC_COMMIT = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
BANKING_COMMIT = "fa9553312b1dba54b1be0aae2d30b13171f898e5"
DATASET_DIRS = {"clinc150": "CLINC150", "banking77_oos": "BANKING77-OOS"}
ASSETS = Path(SEALED_MOUNT) / "e2" / "assets"
ER1_SOURCE = Path(__file__).resolve().parents[1] / "third_party" / "er1" / "src"
TRAIN_IMAGE_TAG = "nvidia/cuda:13.2.1-cudnn-runtime-ubuntu24.04"
TORCH_PIN = "torch==2.15.0"
TRANSFORMERS_PIN = "transformers==4.57.1"


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    backbone: str
    arm: str
    seed: int

    @property
    def group(self) -> str:
        dataset = "banking77" if self.dataset == "banking77_oos" else self.dataset
        return f"{dataset}_{self.backbone}"

    @property
    def leaf(self) -> str:
        return f"{self.arm}_seed{self.seed}"

    @property
    def name(self) -> str:
        return f"{self.group}/{self.leaf}"


def plan_runs() -> list[RunSpec]:
    return [RunSpec(dataset, backbone, arm, seed)
            for dataset in DATASETS for backbone in BACKBONES
            for arm in ARMS for seed in SEEDS]


def _asset_manifest() -> dict:
    return {
        "clinc_commit": CLINC_COMMIT,
        "banking_commit": BANKING_COMMIT,
        "models": {name: {"repository": repo, "commit": commit}
                   for name, (repo, commit) in BACKBONES.items()},
    }


def _check_assets() -> None:
    manifest_path = ASSETS / "manifest.json"
    if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != _asset_manifest():
        raise RuntimeError("E2 assets are missing or have different revisions; run prepare")
    files = [ASSETS / "datasets" / "CLINC150" / "data_oos_plus.json",
             ASSETS / "datasets" / "BANKING77-OOS" / "train" / "seq.in"]
    files += [ASSETS / "models" / name / "model.safetensors" for name in BACKBONES]
    if any(not path.is_file() for path in files):
        raise RuntimeError("E2 assets are incomplete in the sealed volume")


def _clone_commit(url: str, commit: str, destination: Path) -> None:
    subprocess.run(["git", "init", "-q", str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "remote", "add", "origin", url], check=True)
    subprocess.run(["git", "-C", str(destination), "fetch", "-q", "--depth=1", "origin", commit], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "-q", "--detach", "FETCH_HEAD"], check=True)
    actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise RuntimeError(f"Dataset source changed: expected {commit}, got {actual}")


prep_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("huggingface_hub==0.36.0")
    .env(settings.container_env())
    .add_local_python_source("modal_jobs")
)


@app.function(image=prep_image, volumes={SEALED_MOUNT: sealed_volume}, cpu=2, memory=4096, timeout=3600)
def prepare_assets() -> dict:
    """Fetch pinned public datasets and model snapshots into sealed storage."""
    from huggingface_hub import snapshot_download

    if ASSETS.exists():
        _check_assets()
        return {"status": "ready", "sources": _asset_manifest()}

    stage = ASSETS.with_name(f".assets-{uuid4().hex}")
    stage.mkdir(parents=True)
    try:
        with tempfile.TemporaryDirectory() as temp:
            clinc, banking = Path(temp) / "clinc", Path(temp) / "banking"
            _clone_commit("https://github.com/clinc/oos-eval.git", CLINC_COMMIT, clinc)
            _clone_commit("https://github.com/jianguoz/Few-Shot-Intent-Detection.git", BANKING_COMMIT, banking)
            clinc_dir = stage / "datasets" / "CLINC150"
            clinc_dir.mkdir(parents=True)
            shutil.copy2(clinc / "data" / "data_oos_plus.json", clinc_dir / "data_oos_plus.json")
            shutil.copytree(banking / "Datasets" / "BANKING77-OOS",
                            stage / "datasets" / "BANKING77-OOS",
                            ignore=shutil.ignore_patterns(".DS_Store"))

        model_files = ["config.json", "model.safetensors", "tokenizer.json",
                       "tokenizer_config.json", "vocab.txt", "vocab.json", "merges.txt"]
        for name, (repo, commit) in BACKBONES.items():
            model_dir = stage / "models" / name
            snapshot_download(repo_id=repo, revision=commit, local_dir=model_dir,
                              allow_patterns=model_files)
            if not (model_dir / "model.safetensors").is_file():
                raise RuntimeError(f"Model weights missing for {name}")

        (stage / "manifest.json").write_text(json.dumps(_asset_manifest(), sort_keys=True, indent=2))
        stage.rename(ASSETS)
        sealed_volume.commit()
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return {"status": "prepared", "sources": _asset_manifest()}


train_image = (
    modal.Image.from_registry(TRAIN_IMAGE_TAG, add_python="3.12")
    .pip_install(TORCH_PIN, TRANSFORMERS_PIN, "numpy==1.26.4")
    .env({**settings.container_env(), "PYTHONPATH": "/opt/er1/src",
          "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
)
if ER1_SOURCE.is_dir():
    # The source is baked into the image when launching locally. Remote imports
    # already have it at /opt/er1/src and need no host-side submodule path.
    train_image = train_image.add_local_dir(ER1_SOURCE, "/opt/er1/src", copy=True)
train_image = train_image.add_local_python_source("modal_jobs")


def _recipe(spec: RunSpec) -> dict:
    repo, revision = BACKBONES[spec.backbone]
    return {**asdict(spec), "model_repository": repo, "model_commit": revision,
            "er1_commit": ER1_COMMIT, "clinc_commit": CLINC_COMMIT,
            "banking_commit": BANKING_COMMIT, "epochs": 5, "batch_size": 32,
            "max_length": 64, "feature_dim": 256,
            "image": TRAIN_IMAGE_TAG, "torch": TORCH_PIN,
            "transformers": TRANSFORMERS_PIN}


def _fingerprint(recipe: dict) -> str:
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def publish_open_cache(sealed_dir: Path, open_dir: Path, recipe: dict) -> None:
    """Allowlist arrays so test splits and validation labels cannot reach agents."""
    import numpy as np

    keys = {f"{split}__{field}" for split in ("train", "val", "oos_val")
            for field in ("logits", "feats", "pooled")}
    keys.add("train__labels")
    if open_dir.exists():
        manifest = open_dir / "manifest.json"
        cache_path = open_dir / "cache.npz"
        if manifest.is_file() and cache_path.is_file():
            if json.loads(manifest.read_text()).get("recipe_hash") == _fingerprint(recipe):
                with np.load(cache_path, allow_pickle=False) as existing:
                    if set(existing.files) == keys:
                        return
        raise RuntimeError(f"Open cache already exists with different or incomplete contents: {open_dir}")

    stage = open_dir.with_name(f".{open_dir.name}.{uuid4().hex}.tmp")
    stage.mkdir(parents=True)
    try:
        with np.load(sealed_dir / "cache.npz", allow_pickle=False) as cache, \
                np.load(sealed_dir / "pooled.npz", allow_pickle=False) as pooled:
            arrays = {key: (pooled[key] if key.endswith("__pooled") else cache[key]) for key in keys}
            if any(not np.isfinite(array).all() for array in arrays.values()):
                raise RuntimeError("Non-finite values in training or validation cache")
            np.savez_compressed(stage / "cache.npz", **arrays)
        (stage / "manifest.json").write_text(json.dumps({"run": recipe,
                                "recipe_hash": _fingerprint(recipe)}, sort_keys=True, indent=2))
        stage.rename(open_dir)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _verify_gpu() -> None:
    import torch

    cuda = torch.version.cuda
    if not torch.cuda.is_available() or not cuda or tuple(map(int, cuda.split(".")[:2])) < (13, 1):
        raise RuntimeError("B300 training needs a working CUDA 13.1+ PyTorch build")
    name = torch.cuda.get_device_name(0)
    if "B300" not in name.upper():
        raise RuntimeError(f"Expected a B300, got {name}")
    x = torch.ones((32, 32), device="cuda")
    torch.mm(x, x)
    torch.cuda.synchronize()


@app.function(image=train_image,
              volumes={OPEN_MOUNT: open_volume, SEALED_MOUNT: sealed_volume},
              **TRAIN40_MODAL_OPTIONS)
def train_one(spec_dict: dict) -> dict:
    """Train one ER-1 seed; return status only, never test labels or scores."""
    import torch
    from bert_daedl.config import ExperimentConfig
    from bert_daedl.data import build_dataloaders
    from bert_daedl.models.bert_edl import BertEvidentialClassifier
    from bert_daedl.posthoc.cache import eval_order_loader, extract_split, save_cache
    from bert_daedl.train import train_model
    from bert_daedl.utils.seed import seed_everything

    spec = RunSpec(**spec_dict)
    if spec not in plan_runs():
        raise ValueError(f"Invalid E2 run: {spec}")
    _verify_gpu()
    sealed_volume.reload()
    _check_assets()
    recipe = _recipe(spec)
    recipe_hash = _fingerprint(recipe)
    sealed_dir = Path(SEALED_MOUNT) / spec.name
    open_dir = Path(OPEN_MOUNT) / spec.name

    if sealed_dir.exists():
        marker = sealed_dir / "run_complete.json"
        if not marker.is_file() or json.loads(marker.read_text()).get("recipe_hash") != recipe_hash:
            raise RuntimeError(f"Sealed cache already exists with different or incomplete contents: {sealed_dir}")
        status = "already_trained"
    else:
        if open_dir.exists():
            raise RuntimeError(f"Open cache exists without a sealed run: {open_dir}")
        sealed_dir.parent.mkdir(parents=True, exist_ok=True)
        stage = sealed_dir.with_name(f".{spec.leaf}.{uuid4().hex}.tmp")
        stage.mkdir()
        try:
            cfg = ExperimentConfig(dataset=spec.dataset,
                                   data_dir=str(ASSETS / "datasets" / DATASET_DIRS[spec.dataset]),
                                   model_name=str(ASSETS / "models" / spec.backbone),
                                   arm=spec.arm, seed=spec.seed, device="cuda",
                                   num_workers=4, output_dir=str(stage))
            seed_everything(spec.seed)
            tr, val, test, oos_val, oos_test, meta = build_dataloaders(cfg.dataset, cfg, cfg.data_dir)
            model = BertEvidentialClassifier(cfg.model_name, meta["num_classes"],
                                             cfg.feat_dim, cfg.sn_bottleneck,
                                             cfg.bottleneck_activation, cfg.pooling, cfg.dropout)
            model, best_val_acc = train_model(model, tr, val, oos_val, cfg, meta["num_classes"])
            ordered_train = eval_order_loader(tr, batch_size=cfg.eval_batch_size,
                                              num_workers=cfg.num_workers)
            arrays = {name: extract_split(model, loader, cfg.device, with_pooled=True)
                      for name, loader in (("train", ordered_train), ("val", val), ("test", test),
                                           ("oos_val", oos_val), ("oos_test", oos_test))}
            save_cache(stage, arrays, {"arm": spec.arm, "seed": spec.seed,
                       "dataset": spec.dataset, "model_name": BACKBONES[spec.backbone][0],
                       "model_commit": BACKBONES[spec.backbone][1], "er1_commit": ER1_COMMIT,
                       "best_val_acc": best_val_acc, "recipe_hash": recipe_hash,
                       "label_names": meta.get("id2label")})
            torch.save({key: value.cpu() for key, value in model.state_dict().items()}, stage / "model.pt")
            (stage / "run_complete.json").write_text(json.dumps({"recipe_hash": recipe_hash,
                                      "best_val_acc": best_val_acc}, sort_keys=True))
            stage.rename(sealed_dir)
            sealed_volume.commit()
        except BaseException:
            if stage.exists():
                shutil.rmtree(stage)
            raise
        status = "trained"

    publish_open_cache(sealed_dir, open_dir, recipe)
    open_volume.commit()
    return {"run": spec.name, "status": status, "open_cache": str(open_dir)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "prepare", "run"))
    parser.add_argument("--all", action="store_true", help="Run all 40 configurations")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--backbone", choices=tuple(BACKBONES))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args(argv)

    runs = plan_runs()
    if args.command == "plan":
        for spec in runs:
            print(spec.name)
        print(f"{len(runs)} runs; at most {TRAIN40_MODAL_OPTIONS['max_containers']} B300s in parallel")
        return 0

    if args.command == "run":
        filters = ("dataset", "backbone", "arm", "seed")
        if not args.all and any(getattr(args, field) is None for field in filters):
            parser.error("Select one run with all four filters, or pass --all")
        if args.all and any(getattr(args, field) is not None for field in filters):
            parser.error("--all cannot be combined with individual run filters")
        if not args.all:
            runs = [spec for spec in runs if all(getattr(spec, field) == getattr(args, field)
                    for field in filters)]
        if not ER1_SOURCE.is_dir():
            parser.error("The ER-1 submodule is missing; run make bootstrap")
        actual = subprocess.check_output(["git", "-C", str(ER1_SOURCE.parent), "rev-parse", "HEAD"], text=True).strip()
        if actual != ER1_COMMIT:
            parser.error("The ER-1 submodule is not at its pinned commit; run make bootstrap")

    with modal.enable_output():
        with app.run(environment_name=settings.environment):
            prep = prepare_assets.remote()
            print(json.dumps(prep, sort_keys=True))
            if args.command == "run":
                for result in train_one.map((asdict(spec) for spec in runs), order_outputs=False):
                    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
