"""E4 sealed test scoring after a published decision freeze.

The trusted function mounts /sealed and reads test arrays through SealedStore.
Each proposal runs in a fresh network-blocked Sandbox with no volume or secret.
Only a status returns; per-item test scores stay in /sealed/scores.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
from pathlib import Path
from uuid import uuid4

import modal
import numpy as np

from modal_jobs.app import sealed_volume, settings
from modal_jobs.config import SEALED_MOUNT
from modal_jobs.sandbox import (MAX_SOURCE_BYTES, _score_in_sandbox,
                                runner_image, sandbox_image)
from modal_jobs.train40 import RunSpec, plan_runs
from skeptic import api
from skeptic.sandbox import pack_input
from skeptic.store import (GuardError, SealedStore, verify_published_freeze,
                           verify_remote_proposal)

REPO = Path(__file__).resolve().parents[1]
seal_app = modal.App(f"{settings.app_name}-e4")


def _cell(dataset: str, backbone: str) -> str:
    return f"{'banking77_oos' if dataset == 'banking77' else dataset}/{backbone}"


@seal_app.function(image=sandbox_image, cpu=0.25, memory=128, timeout=30,
                   block_network=True, restrict_modal_access=True)
def sandbox_image_ready() -> None:
    """Register the worker image before a sealed scoring call starts."""


def prepare_test_cell(store: SealedStore, spec: RunSpec) -> tuple[bytes, bytes, np.ndarray, np.ndarray, int, int]:
    """Create anonymous full/subset payloads without a test label or membership mask."""
    if spec not in plan_runs():
        raise ValueError("Unknown E2 run")
    train = {field: store.read_array(spec.name, "train", field)
             for field in ("logits", "feats", "pooled", "labels")}
    inside = {field: store.read_array(spec.name, "test", field)
              for field in ("logits", "feats", "pooled")}
    outside = {field: store.read_array(spec.name, "oos_test", field)
               for field in ("logits", "feats", "pooled")}
    n_in, n_oos = len(inside["logits"]), len(outside["logits"])
    k = 150 if spec.dataset == "clinc150" else 50
    if n_in < 2 or n_oos < 2 or len(train["logits"]) < 2:
        raise GuardError("Sealed cache has an empty or insufficient split")
    for part in (train, inside, outside):
        n = len(part["logits"])
        if any(np.asarray(part[field]).ndim != 2 or len(part[field]) != n
               for field in ("logits", "feats", "pooled")) \
                or part["logits"].shape[1] != k or part["feats"].shape[1] != 256 \
                or part["pooled"].shape[1] != 768 \
                or any(not np.isfinite(part[field]).all()
                       for field in ("logits", "feats", "pooled")):
            raise GuardError("Sealed cache shape or values are invalid")
    labels = np.asarray(train["labels"])
    if labels.shape != (len(train["logits"]),) or not np.issubdtype(labels.dtype, np.integer) \
            or (labels < 0).any() or (labels >= k).any():
        raise GuardError("Sealed training labels are invalid")

    rng = np.random.default_rng(secrets.randbits(128))
    permutation = rng.permutation(n_in + n_oos)
    eval_data = {field: np.concatenate((inside[field], outside[field]))[permutation]
                 for field in ("logits", "feats", "pooled")}
    subset_idx = np.sort(rng.choice(n_in + n_oos,
                                    size=max(2, math.ceil(api.SUBSET_FRAC * (n_in + n_oos))),
                                    replace=False))
    subset = {field: value[subset_idx] for field, value in eval_data.items()}
    return pack_input(train, eval_data), pack_input(train, subset), permutation, subset_idx, n_in, k


def _existing_score(path: Path, sha: str, commit: str, digest: str) -> bool:
    if not path.exists():
        return False
    if path.is_symlink() or not path.is_file():
        raise GuardError("Existing sealed score path is unsafe")
    try:
        with np.load(path, allow_pickle=False) as saved:
            n_in = np.asarray(saved["n_in"])
            if set(saved.files) != {"scores", "n_in", "proposal_sha", "decision_commit", "decision_digest"} \
                    or n_in.shape != () or not np.issubdtype(n_in.dtype, np.integer) \
                    or str(saved["proposal_sha"]) != sha \
                    or str(saved["decision_commit"]) != commit \
                    or str(saved["decision_digest"]) != digest \
                    or saved["scores"].ndim != 2 or saved["scores"].shape[0] != 5 \
                    or not np.isfinite(saved["scores"]).all() \
                    or not 0 < int(saved["n_in"]) < saved["scores"].shape[1]:
                raise GuardError("Existing sealed score has different provenance or invalid data")
    except (OSError, ValueError, KeyError) as exc:
        raise GuardError("Existing sealed score is unreadable") from exc
    return True


@seal_app.function(image=runner_image, volumes={SEALED_MOUNT: sealed_volume},
                   cpu=2, memory=8192, timeout=900, max_containers=4)
def score_sealed(text: str, dataset: str, backbone: str, arm: str, commit: str,
                 digest: str, worker_image_id: str) -> dict:
    """Run five isolated seed scores and return status only, never test metrics."""
    if len(text.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("Proposal source is too large")
    header = api.static_check(text)
    if header["ARM"] != arm:
        raise ValueError("Proposal arm does not match the target cell")
    specs = [RunSpec(dataset, backbone, arm, seed) for seed in range(5)]
    if any(spec not in plan_runs() for spec in specs):
        raise ValueError("Unknown E2 cell")

    sealed_volume.reload()
    store = SealedStore(SEALED_MOUNT, REPO)
    store.unseal_remote(commit, digest)
    sha = api.proposal_sha(text)
    cell = _cell(dataset, backbone)
    verify_remote_proposal(commit, cell, sha)
    group = f"{dataset}_{backbone}"
    target = Path(SEALED_MOUNT) / "scores" / group / f"{sha}.npz"
    if any(p.is_symlink() for p in (target.parent.parent, target.parent, target)):
        raise GuardError("Sealed score paths cannot contain symlinks")
    if _existing_score(target, sha, commit, digest):
        return {"sha": sha, "cell": cell, "status": "already_sealed"}

    owner = modal.App.lookup(settings.app_name, environment_name=settings.environment)
    image = modal.Image.from_id(worker_image_id)
    per_seed = []
    n_in = None
    total_n = None
    for spec in specs:
        full, subset, permutation, subset_idx, current_n_in, k = prepare_test_cell(store, spec)
        if n_in is None:
            n_in, total_n = current_n_in, len(permutation)
        elif n_in != current_n_in or len(permutation) != total_n:
            raise GuardError("Sealed test items are misaligned across seeds")
        scores, _ = _score_in_sandbox(owner, image, text, sha, full, total_n, k)
        sub_scores, _ = _score_in_sandbox(owner, image, text, sha, subset, len(subset_idx), k,
                                          subset=True)
        if not np.allclose(sub_scores, scores[subset_idx], rtol=1e-6, atol=1e-8):
            raise api.Malformed("subset_dependent")
        original = np.empty_like(scores)
        original[permutation] = scores
        per_seed.append(original.astype(np.float32))
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_name(f".{sha}.{uuid4().hex}.tmp.npz")
    try:
        np.savez_compressed(stage, scores=np.stack(per_seed), n_in=n_in, proposal_sha=sha,
                            decision_commit=commit, decision_digest=digest)
        if target.exists():
            raise GuardError("Sealed score appeared during scoring; refusing overwrite")
        os.link(stage, target)
    finally:
        stage.unlink(missing_ok=True)
    sealed_volume.commit()
    return {"sha": sha, "cell": cell, "status": "sealed"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--proposal", required=True, type=Path,
                        help="content-hashed proposal file from proposals/")
    parser.add_argument("--dataset", choices=("clinc150", "banking77"), required=True)
    parser.add_argument("--backbone", choices=("bert", "roberta"), required=True)
    args = parser.parse_args(argv)
    text = args.proposal.read_text(encoding="utf-8")
    sha = api.proposal_sha(text)
    if args.proposal.stem != sha or args.proposal.parent.resolve() != (REPO / "proposals").resolve():
        raise ValueError("Proposal must be the content-hashed file in this repository")
    arm = api.static_check(text)["ARM"]
    target = {"sha": sha, "cell": _cell(args.dataset, args.backbone),
              "dataset": args.dataset, "backbone": args.backbone, "arm": arm}
    if args.command == "plan":
        print(json.dumps(target, sort_keys=True))
        return 0
    permit = verify_published_freeze(REPO)
    with modal.enable_output():
        with seal_app.run(environment_name=settings.environment):
            result = score_sealed.remote(text, args.dataset, args.backbone, arm,
                                         permit.commit, permit.digest, sandbox_image.object_id)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
