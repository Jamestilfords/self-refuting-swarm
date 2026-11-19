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
from modal_jobs.authorization import accept_freeze_receipt, freeze_receipt
from modal_jobs.config import SEALED_MOUNT
from modal_jobs.sandbox import (CELL_JOB_TIMEOUT, MAX_SOURCE_BYTES, _score_in_sandbox, _spec,
                                runner_image, sandbox_image)
from modal_jobs.train40 import RunSpec, plan_runs
from skeptic import api
from skeptic.sandbox import pack_input
from skeptic.store import (GuardError, SealedStore, _manifest_paths, verify_published_freeze,
                           verify_remote_proposal, verify_remote_proposals)

REPO = Path(__file__).resolve().parents[1]
seal_app = modal.App(f"{settings.app_name}-e4")
MAX_BATCH_SIZE = 25
MAX_CONCURRENT_BATCHES = 40
FREEZE_VERIFICATION_ALLOWANCE = 30 * 60
SEALED_CELL_TIMEOUT = CELL_JOB_TIMEOUT + FREEZE_VERIFICATION_ALLOWANCE
SEALED_BATCH_TIMEOUT = MAX_BATCH_SIZE * CELL_JOB_TIMEOUT + FREEZE_VERIFICATION_ALLOWANCE


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
                   cpu=2, memory=8192, timeout=SEALED_CELL_TIMEOUT, max_containers=4)
def score_sealed(text: str, dataset: str, backbone: str, arm: str, commit: str,
                 digest: str, worker_image_id: str) -> dict:
    """Run five isolated seed scores and return status only, never test metrics."""
    if len(text.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("Proposal source is too large")
    header = api.static_check(text)
    if header["ARM"] != arm:
        raise ValueError("Proposal arm does not match the target cell")
    specs = [_spec(dataset, backbone, arm, seed) for seed in range(5)]

    sealed_volume.reload()
    store = SealedStore(SEALED_MOUNT, REPO)
    store.unseal_remote(commit, digest)
    sha = api.proposal_sha(text)
    cell = _cell(dataset, backbone)
    verify_remote_proposal(commit, cell, sha)
    return _score_authorized(text, specs, store, commit, digest, worker_image_id)


def _score_authorized(text, specs, store, commit, digest, worker_image_id):
    """Trusted implementation; caller must verify freeze and proposal membership first."""
    sha = api.proposal_sha(text)
    cell = _cell(specs[0].dataset, specs[0].backbone)
    group = specs[0].group
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


@seal_app.function(image=runner_image, volumes={SEALED_MOUNT: sealed_volume},
                   cpu=2, memory=8192, timeout=SEALED_BATCH_TIMEOUT,
                   max_containers=MAX_CONCURRENT_BATCHES)
def score_sealed_batch(jobs: list[dict], commit: str, digest: str, worker_image_id: str,
                       authorization=None) -> list[dict]:
    """Verify once per bounded chunk; return statuses, never test metrics or errors' text."""
    if not jobs or len(jobs) > MAX_BATCH_SIZE:
        raise ValueError("Invalid sealed batch size")
    targets, specs_by_job = [], []
    for job in jobs:
        text, sha, cell = job["text"], job["sha"], job["cell"]
        if len(text.encode("utf-8")) > MAX_SOURCE_BYTES or api.proposal_sha(text) != sha:
            raise ValueError("Invalid proposal source")
        arm = api.static_check(text)["ARM"]
        if arm != job["arm"] or cell != _cell(job["dataset"], job["backbone"]):
            raise ValueError("Invalid sealed job identity")
        specs_by_job.append([_spec(job["dataset"], job["backbone"], arm, seed) for seed in range(5)])
        targets.append((cell, sha))
    if len(targets) != len(set(targets)):
        raise ValueError("Duplicate sealed job")
    sealed_volume.reload()
    store = SealedStore(SEALED_MOUNT, REPO)
    if authorization is None:
        # Standalone calls retain independent remote verification.
        store.unseal_remote(commit, digest)
        verify_remote_proposals(commit, targets)
    else:
        _authorize_batch(store, authorization, commit, digest, targets)
    results = []
    for job, specs in zip(jobs, specs_by_job):
        try:
            result = _score_authorized(job["text"], specs, store, commit, digest, worker_image_id)
        except Exception as exc:
            # Exception text can contain generated-code output or sealed data.
            result = {"sha": job["sha"], "cell": job["cell"], "status": "failed",
                      "error_type": type(exc).__name__}
        results.append(result)
    return results


def _read_proposal(path: Path) -> tuple[str, str, str]:
    if path.is_symlink() or path.parent.is_symlink() or not path.is_file() \
            or path.parent.resolve() != (REPO / "proposals").resolve():
        raise ValueError("Proposal must be the content-hashed file in this repository")
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("Proposal source is too large")
    text = path.read_text(encoding="utf-8")
    sha = api.proposal_sha(text)
    if path.stem != sha:
        raise ValueError("Proposal filename does not match its content hash")
    return text, sha, api.static_check(text)["ARM"]


def plan_jobs(proposal: Path | None, dataset: str | None, backbone: str | None) -> list[dict]:
    """Offline plan: all manifest proposals, or one explicit proposal/cell."""
    dataset = "banking77_oos" if dataset == "banking77" else dataset
    if proposal is not None:
        if dataset is None or backbone is None:
            raise ValueError("--proposal requires --dataset and --backbone")
        selected = [(_cell(dataset, backbone), proposal)]
    else:
        _, proposals, references = _manifest_paths(REPO)
        selected = [(cell, REPO / "proposals" / f"{sha}.py")
                    for cell in sorted(proposals)
                    if (dataset is None or cell.split("/")[0] == dataset)
                    and (backbone is None or cell.split("/")[1] == backbone)
                    for sha in sorted(set(proposals[cell]) | set(references[cell].values()))]
    jobs = []
    for cell, path in selected:
        text, sha, arm = _read_proposal(path)
        ds, bb = cell.split("/")
        _spec(ds, bb, arm, 0)
        jobs.append({"sha": sha, "cell": cell, "dataset": ds,
                     "backbone": bb, "arm": arm, "text": text})
    if not jobs:
        raise ValueError("No matching proposals in the decision manifest")
    return jobs


def _authorize_batch(store, authorization, commit, digest, targets):
    """Accept only the trusted coordinator's workspace-authenticated run receipt.

    This object is never passed into proposal sandboxes. Like the sealed volume,
    it is a capability held only by trusted Modal functions and the coordinator.
    """
    store._permit = None
    permit = accept_freeze_receipt(authorization, commit, digest, "seal")
    allowed = {tuple(item) for item in authorization["freeze"].get("targets", [])}
    if not set(targets).issubset(allowed):
        raise GuardError("Proposal is not in the authorized decision set")
    store._permit = permit


def _collect_batches(jobs, permit, worker_image_id, batch_size, report, authorization=None):
    """Append status-only progress; remote sealed artifacts are the resume authority."""
    chunks = [jobs[i:i + batch_size] for i in range(0, len(jobs), batch_size)]
    counts = {"sealed": 0, "already_sealed": 0, "failed": 0}
    report.parent.mkdir(parents=True, exist_ok=True)
    if report.is_symlink():
        raise ValueError("Status report cannot be a symlink")
    with report.open("a", encoding="utf-8") as handle:
        results = iter(score_sealed_batch.starmap(
            [(chunk, permit.commit, permit.digest, worker_image_id, authorization) for chunk in chunks],
            order_outputs=True, return_exceptions=True))
        for chunk in chunks:
            try:
                result = next(results)
                if isinstance(result, BaseException):
                    raise RuntimeError("Remote batch failed")
                expected = {(job["sha"], job["cell"]) for job in chunk}
                if not isinstance(result, list) or len(result) != len(chunk) \
                        or any(not isinstance(row, dict) or row.get("status") not in counts
                               or set(row) != ({"sha", "cell", "status", "error_type"}
                                   if row["status"] == "failed" else {"sha", "cell", "status"})
                               for row in result) \
                        or {(row["sha"], row["cell"]) for row in result} != expected:
                    raise ValueError("Invalid remote status response")
                # Never forward remote error text, even in a nominally status-only response.
                result = [{**{k: row[k] for k in ("sha", "cell", "status")},
                           **({"error_type": "ScoringError"} if row["status"] == "failed" else {})}
                          for row in result]
            except Exception:
                result = [{"sha": job["sha"], "cell": job["cell"], "status": "failed",
                           "error_type": "BatchError"} for job in chunk]
            for row in result:
                record = {"schema_version": 1, "decision_commit": permit.commit,
                          "decision_digest": permit.digest, **row}
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                counts[row["status"]] += 1
            handle.flush()
            print(json.dumps({"completed": sum(counts.values()), "total": len(jobs), **counts}), flush=True)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("plan", "run"))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--proposal", type=Path,
                        help="content-hashed proposal file from proposals/")
    selection.add_argument("--all", action="store_true", help="Score every proposal/cell in the frozen manifest")
    parser.add_argument("--dataset", choices=("clinc150", "banking77", "banking77_oos"))
    parser.add_argument("--backbone", choices=("bert", "roberta"))
    parser.add_argument("--batch-size", type=int, default=MAX_BATCH_SIZE,
                        help=f"Proposal/cell jobs per remote chunk (1-{MAX_BATCH_SIZE})")
    parser.add_argument("--report", type=Path, default=REPO / "runs/seal/status.jsonl",
                        help="Append-only batch status report; never contains test metrics")
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= MAX_BATCH_SIZE:
        parser.error(f"--batch-size must be between 1 and {MAX_BATCH_SIZE}")
    try:
        jobs = plan_jobs(args.proposal, args.dataset, args.backbone)
    except (ValueError, OSError, GuardError) as exc:
        parser.error(str(exc))
    if args.command == "plan":
        public = [{k: v for k, v in job.items() if k != "text"} for job in jobs]
        print(json.dumps({"jobs": public, "job_count": len(jobs), "batch_size": args.batch_size,
                          "batches": math.ceil(len(jobs) / args.batch_size),
                          "max_concurrent_batches": MAX_CONCURRENT_BATCHES,
                          "max_sandbox_executions": 10 * len(jobs), "freeze_verified": False}
                         if args.all else public[0], sort_keys=True))
        return 0
    permit = verify_published_freeze(REPO)
    with modal.enable_output():
        with seal_app.run(environment_name=settings.environment):
            if args.all:
                # The complete published freeze was verified above, once. Workers
                # read this ephemeral capability instead of downloading 4 GiB each.
                with modal.Dict.ephemeral(environment_name=settings.environment) as authorization:
                    authorization["freeze"] = freeze_receipt(
                        permit, "seal", targets=[(j["cell"], j["sha"]) for j in jobs])
                    counts = _collect_batches(jobs, permit, sandbox_image.object_id, args.batch_size,
                                              args.report, authorization)
                return int(counts["failed"] > 0)
            job = jobs[0]
            result = score_sealed.remote(job["text"], job["dataset"], job["backbone"], job["arm"],
                                         permit.commit, permit.digest, sandbox_image.object_id)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
