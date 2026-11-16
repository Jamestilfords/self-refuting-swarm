"""Isolated per-item validation scorer for E5 replay.

The trusted function mounts only /open. Proposal code runs in fresh Sandboxes
with no mounted volume, network, or secret. New artifacts go to
/open/scores_isolated, leaving the earlier score files untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

import modal
import numpy as np

from modal_jobs.app import open_volume, sealed_volume, settings
from modal_jobs.config import OPEN_MOUNT, SEALED_MOUNT
from modal_jobs.sandbox import (MAX_SOURCE_BYTES, _score_in_sandbox, _spec,
                                prepare_cell, runner_image, sandbox_image)
from skeptic import api, runner
from skeptic.score_artifacts import CELLS, ScoreArtifact, auroc

REPO = Path(__file__).resolve().parents[1]
stream_app = modal.App(f"{settings.app_name}-stream")


def _snapshot(src: Path) -> Path:
    dst = Path(tempfile.mkdtemp(prefix="skeptic_proposals_"))
    for path in sorted(src.iterdir()):
        if path.suffix in (".py", ".json"):
            shutil.copy2(path, dst / path.name)
    return dst


PROPOSALS = _snapshot(REPO / "proposals") if (REPO / "proposals").is_dir() else REPO / "proposals"


@stream_app.function(image=sandbox_image, cpu=0.25, memory=128, timeout=30,
                     block_network=True, restrict_modal_access=True)
def sandbox_image_ready() -> None:
    """Register the worker image before scoring calls start."""


@stream_app.function(image=runner_image, cpu=2, memory=4096, timeout=1800,
                     volumes={OPEN_MOUNT: open_volume.with_mount_options(read_only=True),
                              SEALED_MOUNT: sealed_volume.with_mount_options(read_only=True)})
def calibrate_cell(task: dict) -> dict:
    """Compute G5 from stored scores and validation labels; execute no proposal."""
    open_volume.reload()
    sealed_volume.reload()
    rows = runner.validation_calibration(OPEN_MOUNT, SEALED_MOUNT, task["group"],
                                         task["items"], source="isolated")
    return {"cell": task["cell"], "rows": rows}


def _metrics(scores: np.ndarray, n_in: int) -> dict:
    return {"auroc": [round(auroc(seed[:n_in], seed[n_in:]), 6) for seed in scores],
            "fpr95": [round(runner.fpr95(seed[:n_in], seed[n_in:]), 6) for seed in scores]}


def _existing(path: Path, sha: str) -> tuple[np.ndarray, int, float, bool] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Existing isolated score path is unsafe")
    with np.load(path, allow_pickle=False) as cache:
        keys = set(cache.files)
        base = {"scores", "n_in", "runtime_s", "scorer", "proposal_sha"}
        if keys not in (base, base | {"pred", "conf"}) \
                or str(cache["scorer"]) != "isolated-e3-v1" \
                or str(cache["proposal_sha"]) != sha:
            raise RuntimeError("Existing isolated score provenance is invalid")
        n_in, runtime = np.asarray(cache["n_in"]), np.asarray(cache["runtime_s"])
        if n_in.shape != () or not np.issubdtype(n_in.dtype, np.integer) \
                or runtime.shape != () or not np.isfinite(runtime) or runtime < 0:
            raise RuntimeError("Existing isolated score metadata is invalid")
        artifact = ScoreArtifact(cache["scores"], int(n_in),
                                 cache["pred"] if "pred" in keys else None,
                                 cache["conf"] if "conf" in keys else None)
        return artifact.scores, artifact.n_in, float(cache["runtime_s"]), artifact.pred is not None


@stream_app.function(image=runner_image, volumes={OPEN_MOUNT: open_volume},
                     cpu=2, memory=8192, timeout=900, max_containers=4)
def score_one(task: dict, worker_image_id: str) -> dict:
    """Write a five-seed artifact and return only validation summary fields."""
    cell, sha, text, arm = (task[key] for key in ("cell", "sha", "text", "arm"))
    if cell not in CELLS or len(text.encode("utf-8")) > MAX_SOURCE_BYTES or api.proposal_sha(text) != sha:
        raise ValueError("Invalid cell or proposal source")
    row = {"sha": sha, "status": "ok", "family": None, "arm": arm,
           "parents": [], "source": "", "memory_used": [],
           "runtime_s": None, "val": None, "has_probs": False}
    try:
        header = api.static_check(text)
        if header["ARM"] != arm:
            raise api.Malformed("arm_mismatch")
        row.update(family=header["FAMILY"], parents=header["PARENTS"],
                   source=header["SOURCE"], memory_used=header["MEMORY_USED"])
        dataset, backbone = cell.split("/")
        group = CELLS[cell]
        open_volume.reload()
        target = Path(OPEN_MOUNT) / "scores_isolated" / group / f"{sha}.npz"
        existing = _existing(target, sha)
        if existing is not None:
            scores, n_in, runtime, has_probs = existing
            row.update(val=_metrics(scores, n_in), runtime_s=runtime, has_probs=has_probs)
            return {"cell": cell, "rows": [row]}

        owner = modal.App.lookup(settings.app_name, environment_name=settings.environment)
        image = modal.Image.from_id(worker_image_id)
        all_scores, all_pred, all_conf, runtimes = [], [], [], []
        n_in = n_total = has_probs = None
        for seed in range(5):
            spec = _spec(dataset, backbone, arm, seed)
            full, subset, membership, subset_idx, k, permutation = prepare_cell(
                Path(OPEN_MOUNT), spec, return_permutation=True)
            if n_in is None:
                n_in, n_total = int((~membership).sum()), len(membership)
            elif n_in != int((~membership).sum()) or n_total != len(membership):
                raise RuntimeError("Validation items differ across seeds")
            scores, probs, runtime = _score_in_sandbox(owner, image, text, sha, full,
                                                        n_total, k, return_probs=True)
            subset_scores, subset_probs, _ = _score_in_sandbox(
                owner, image, text, sha, subset, len(subset_idx), k,
                subset=True, return_probs=True)
            if not np.allclose(subset_scores, scores[subset_idx], rtol=1e-6, atol=1e-8):
                raise api.Malformed("subset_dependent")
            if (probs is None) != (subset_probs is None) or (probs is not None and
                    not np.allclose(subset_probs, probs[subset_idx], rtol=1e-6, atol=1e-8)):
                raise api.Malformed("subset_dependent")
            if has_probs is None:
                has_probs = probs is not None
            elif has_probs != (probs is not None):
                raise api.Malformed("probs", "Probability output changed across seeds")
            original = np.empty_like(scores)
            original[permutation] = scores
            all_scores.append(original.astype(np.float32))
            if probs is not None:
                original_probs = np.empty_like(probs)
                original_probs[permutation] = probs
                all_pred.append(original_probs.argmax(1).astype(np.int16))
                all_conf.append(original_probs.max(1).astype(np.float32))
            runtimes.append(runtime)

        packed = {"scores": np.stack(all_scores), "n_in": n_in,
                  "runtime_s": max(runtimes), "scorer": "isolated-e3-v1", "proposal_sha": sha}
        if has_probs:
            packed.update(pred=np.stack(all_pred), conf=np.stack(all_conf))
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = target.with_name(f".{sha}.{uuid4().hex}.tmp.npz")
        try:
            np.savez_compressed(stage, **packed)
            if target.exists():
                raise RuntimeError("Isolated score appeared during scoring; refusing overwrite")
            os.link(stage, target)
        finally:
            stage.unlink(missing_ok=True)
        open_volume.commit()
        row.update(val=_metrics(packed["scores"], n_in), runtime_s=max(runtimes),
                   has_probs=bool(has_probs))
    except api.Malformed as exc:
        row.update(status=exc.reason, detail=exc.detail[:200])
    except Exception as exc:
        row.update(status="runner_error", detail=f"{type(exc).__name__}: {exc}"[:200])
    return {"cell": cell, "rows": [row]}


def _collect(tasks: list[dict], worker_image_id: str) -> dict:
    rows = {}
    failures = 0
    for result in score_one.map(tasks, [worker_image_id] * len(tasks),
                                order_outputs=False, return_exceptions=True):
        if isinstance(result, BaseException):
            failures += 1
            print(f"[validation] batch failed: {type(result).__name__}: {str(result)[:200]}", flush=True)
            continue
        row = result["rows"][0]
        rows[(row["sha"], result["cell"])] = row
    print(f"[validation] {len(rows)} complete; {failures} infrastructure failures", flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--streams", default=None, help="comma-separated: canary,baseline,templated,llm")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N proposals")
    parser.add_argument("--proposals", type=Path, help="Alternate proposal directory, e.g. a new agent run")
    parser.add_argument("--ledger", type=Path, help="Output ledger; required with --proposals to keep runs separate")
    parser.add_argument("--skip-test", action="store_true", help="accepted for old scripts; E4 always scores test separately")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.proposals is not None and args.ledger is None:
        parser.error("--proposals requires an explicit --ledger output path")
    proposal_dir = args.proposals if args.proposals is not None else PROPOSALS
    found = runner.discover(proposal_dir, set(args.streams.split(",")) if args.streams else None)
    tasks = runner.build_tasks(found, args.limit)
    jobs = [{"cell": task["cell"], "arm": task["arm"], "sha": sha,
             "text": (proposal_dir / f"{sha}.py").read_text(encoding="utf-8")}
            for task in tasks for sha in task["shas"]]
    if not jobs:
        parser.error("No matching proposals")
    print(f"{len(jobs)} proposal/cell jobs; isolated scores go to /open/scores_isolated")
    if args.command == "plan":
        return 0
    with modal.enable_output():
        with stream_app.run(environment_name=settings.environment):
            val_rows = _collect(jobs, sandbox_image.object_id)
            cal_tasks = [{"cell": cell, "group": group,
                          "items": [(job["sha"], job["arm"]) for job in jobs
                                    if job["cell"] == cell and
                                    val_rows.get((job["sha"], cell), {}).get("status") == "ok"]}
                         for cell, group in CELLS.items()]
            cal_rows = {}
            for result in calibrate_cell.map([task for task in cal_tasks if task["items"]]):
                cal_rows.update({(row["sha"], result["cell"]): row for row in result["rows"]})
            expected = {key for key, row in val_rows.items() if row["status"] == "ok"}
            if set(cal_rows) != expected:
                raise RuntimeError("Calibration results are incomplete; retry before using the ledger")
    ledger = runner.merge_ledger(found, tasks, val_rows, cal_rows, {})
    partial = args.streams or args.limit
    path = args.ledger or REPO / "runs" / ("ledger_isolated_partial.jsonl" if partial else "ledger_isolated.jsonl")
    runner.write_ledger(ledger, path)
    print(runner.summarize(ledger))
    print(f"wrote {len(ledger)} rows to {path}")
    return int(any(row["status"] in {"infra_error", "runner_error"} for row in ledger))


if __name__ == "__main__":
    raise SystemExit(main())
