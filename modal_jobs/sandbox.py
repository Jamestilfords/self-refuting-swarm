"""E3: run one proposal on one validation cell in disposable CPU Sandboxes.

The trusted Modal function mounts /open read-only, verifies a cache, and sends
only its train arrays and an anonymously shuffled validation batch to the
untrusted Sandbox. The Sandbox has no volumes, network, or secrets. A second,
fresh Sandbox scores a hidden 10% subset for the dependence check. Only a
public summary returns to the local append-only ledger.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import secrets
from pathlib import Path

import modal
import numpy as np
from modal.stream_type import StreamType

from modal_jobs.app import open_volume, settings
from modal_jobs.config import OPEN_MOUNT, SEALED_MOUNT
from modal_jobs.sanity import _allowed_keys, auroc
from modal_jobs.train40 import RunSpec, _fingerprint, _recipe, plan_runs
from skeptic import api, ledger
from skeptic.sandbox import pack_input, unpack_output

MAX_SOURCE_BYTES = 256_000
SANDBOX_TIMEOUT = 90
WORKER_TIMEOUT = 75
ER1_SOURCE = Path(__file__).resolve().parents[1] / "third_party" / "er1" / "src"
ER1_BUNDLE = ER1_SOURCE if ER1_SOURCE.is_dir() else Path("/opt/er1/src")
e3_app = modal.App(f"{settings.app_name}-e3")

runner_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.4", "scipy==1.18.1", "scikit-learn==1.9.1")
    .env(settings.container_env())
    .add_local_python_source("modal_jobs", "skeptic")
    .add_local_dir(ER1_BUNDLE, "/opt/er1/src", copy=True)
)
sandbox_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.4", "scipy==1.18.1", "scikit-learn==1.9.1")
    .env({"SKEPTIC_ER1_SRC": "/opt/er1/src"})
    .add_local_python_source("skeptic")
    .add_local_dir(ER1_BUNDLE, "/opt/er1/src", copy=True)
)


def _spec(dataset: str, backbone: str, arm: str, seed: int) -> RunSpec:
    spec = RunSpec(dataset, backbone, arm, seed)
    if spec not in plan_runs():
        raise ValueError("Unknown E2 cell; choose dataset, backbone, arm and seed from train40 plan")
    return spec


def _base_row(text: str, spec: RunSpec, stream: str) -> dict:
    return {"schema_version": ledger.SCHEMA_VERSION,
            "proposal_sha": api.proposal_sha(text), "run": spec.name,
            "recipe_hash": _fingerprint(_recipe(spec)), "stream": stream,
            "status": "ready", "validation_auroc": None, "runtime_s": None,
            "n_id": None, "n_oos": None}


def preflight(path: Path, spec: RunSpec, stream: str) -> tuple[str, dict]:
    """Read and statically check a file without executing proposal code."""
    text = path.read_text(encoding="utf-8")
    if len(text.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError(f"Proposal exceeds {MAX_SOURCE_BYTES} bytes")
    row = _base_row(text, spec, stream)
    if path.stem != row["proposal_sha"]:
        row["status"] = "name"
    else:
        try:
            header = api.static_check(text)
            if header["ARM"] != spec.arm:
                row["status"] = "arm_mismatch"
        except api.Malformed as exc:
            row["status"] = exc.reason
    ledger.validate_row(row, allow_ready=True)
    return text, row


def prepare_cell(root: Path, spec: RunSpec, *, return_permutation: bool = False):
    """Reject polluted caches and hide ID/OOS membership from the proposal."""
    directory = root / spec.name
    manifest_path, cache_path = directory / "manifest.json", directory / "cache.npz"
    if not manifest_path.is_file() or not cache_path.is_file():
        raise RuntimeError(f"Missing open cache for {spec.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run") != _recipe(spec) or manifest.get("recipe_hash") != _fingerprint(_recipe(spec)):
        raise RuntimeError(f"Open cache recipe mismatch for {spec.name}")
    with np.load(cache_path, allow_pickle=False) as cache:
        if set(cache.files) != _allowed_keys():
            raise RuntimeError(f"Unexpected or missing arrays in open cache for {spec.name}")
        train = {key: np.asarray(cache[f"train__{key}"])
                 for key in ("logits", "feats", "pooled", "labels")}
        val = {key: np.asarray(cache[f"val__{key}"]) for key in ("logits", "feats", "pooled")}
        oos = {key: np.asarray(cache[f"oos_val__{key}"]) for key in ("logits", "feats", "pooled")}
    n_id, n_oos = len(val["logits"]), len(oos["logits"])
    if not n_id or not n_oos or not len(train["logits"]):
        raise RuntimeError("Open cache has an empty split")
    k = train["logits"].shape[1]
    expected_k = 150 if spec.dataset == "clinc150" else 50
    for split in (train, val, oos):
        if any(np.asarray(split[key]).ndim != 2 for key in ("logits", "feats", "pooled")):
            raise RuntimeError("Open cache has a non-matrix feature array")
        if any(len(split[key]) != len(split["logits"]) for key in ("feats", "pooled")):
            raise RuntimeError("Open cache has misaligned arrays")
        if split["logits"].shape[1] != expected_k or split["feats"].shape[1] != 256 \
                or split["pooled"].shape[1] != 768:
            raise RuntimeError("Open cache has unexpected feature dimensions")
        if any(not np.isfinite(split[key]).all() for key in ("logits", "feats", "pooled")):
            raise RuntimeError("Open cache has non-finite arrays")
    if train["labels"].shape != (len(train["logits"]),) or not np.issubdtype(train["labels"].dtype, np.integer):
        raise RuntimeError("Open cache has invalid train labels")
    if k != expected_k or (train["labels"] < 0).any() or (train["labels"] >= k).any():
        raise RuntimeError("Open cache has invalid class labels")

    # Keep the permutation private to the trusted wrapper. The proposal sees
    # one evaluation batch with no split name, membership mask or RNG seed.
    rng = np.random.default_rng(secrets.randbits(128))
    permutation = rng.permutation(n_id + n_oos)
    membership = np.concatenate((np.zeros(n_id, dtype=bool), np.ones(n_oos, dtype=bool)))[permutation]
    eval_data = {key: np.concatenate((val[key], oos[key]))[permutation]
                 for key in ("logits", "feats", "pooled")}
    subset_idx = np.sort(rng.choice(n_id + n_oos,
                                    size=max(2, math.ceil(api.SUBSET_FRAC * (n_id + n_oos))),
                                    replace=False))
    subset = {key: value[subset_idx] for key, value in eval_data.items()}
    result = (pack_input(train, eval_data), pack_input(train, subset), membership, subset_idx, k)
    return (*result, permutation) if return_permutation else result


@e3_app.function(image=sandbox_image, cpu=0.25, memory=128, timeout=30,
                 block_network=True, restrict_modal_access=True)
def sandbox_image_ready() -> None:
    """Register the exact worker image so app.run builds it before E3 calls."""


def sandbox_options(owner: modal.App, image: modal.Image) -> dict:
    """Use an explicit deny-by-default configuration for untrusted code."""
    return {"app": owner, "image": image, "cpu": 2, "memory": 8192,
            "timeout": SANDBOX_TIMEOUT, "runtime": "gvisor", "block_network": True,
            "volumes": {}, "secrets": [], "include_oidc_identity_token": False,
            "env": {"MODAL_TOKEN_ID": None, "MODAL_TOKEN_SECRET": None,
                    "MODAL_PROFILE": None}}


def _score_in_sandbox(owner: modal.App, image: modal.Image, text: str, sha: str, payload: bytes,
                      n: int, k: int, *, subset: bool = False,
                      return_probs: bool = False):
    sandbox = modal.Sandbox.create("sleep", str(SANDBOX_TIMEOUT - 5), **sandbox_options(owner, image))
    try:
        proposal_path = f"/tmp/{sha}.py"
        sandbox.filesystem.write_text(text, proposal_path)
        sandbox.filesystem.write_bytes(payload, "/tmp/input.npz")
        process = sandbox.exec("python", "-m", "skeptic.sandbox", "worker",
                               proposal_path, "/tmp/input.npz", "/tmp/output.npz", "/tmp/status.json",
                               *(["--subset"] if subset else []),
                               stdout=StreamType.DEVNULL, stderr=StreamType.DEVNULL,
                               timeout=WORKER_TIMEOUT)
        try:
            exit_code = process.wait()
        except (modal.exception.ExecTimeoutError, modal.exception.SandboxTimeoutError) as exc:
            raise api.Malformed("timeout") from exc
        if exit_code != 0:
            raise api.Malformed("timeout" if exit_code == -1 else "sandbox_error")
        if sandbox.filesystem.stat("/tmp/status.json").size > 256:
            raise api.Malformed("sandbox_error")
        status = json.loads(sandbox.filesystem.read_bytes("/tmp/status.json"))
        if not isinstance(status, dict) or status.get("status") not in ledger.STATUSES:
            raise api.Malformed("sandbox_error")
        if status["status"] != "ok":
            raise api.Malformed(status["status"])
        if sandbox.filesystem.stat("/tmp/output.npz").size > 8 * n * (k + 1) + 4096:
            raise api.Malformed("sandbox_error")
        return unpack_output(sandbox.filesystem.read_bytes("/tmp/output.npz"), n, k,
                             require_variation=not subset, include_probs=return_probs)
    finally:
        try:
            sandbox.terminate()
        except Exception:
            pass


@e3_app.function(image=runner_image,
                 volumes={OPEN_MOUNT: open_volume.with_mount_options(read_only=True)},
                 cpu=2, memory=8192, timeout=300, max_containers=4)
def score_one(text: str, dataset: str, backbone: str, arm: str, seed: int, worker_image_id: str,
              stream: str = "unknown") -> dict:
    """Trusted wrapper; return a validation summary, never raw scores."""
    spec = _spec(dataset, backbone, arm, seed)
    row = _base_row(text, spec, stream)
    row["status"] = "ok"
    if Path(SEALED_MOUNT).exists():
        raise RuntimeError("The E3 wrapper must not have a sealed mount")
    if len(text.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("Proposal source is too large")
    try:
        header = api.static_check(text)
    except api.Malformed as exc:
        row["status"] = exc.reason
        return ledger.validate_row(row)
    if header["ARM"] != spec.arm:
        row["status"] = "arm_mismatch"
        return ledger.validate_row(row)

    full, subset, membership, subset_idx, k = prepare_cell(Path(OPEN_MOUNT), spec)
    row["n_id"], row["n_oos"] = int((~membership).sum()), int(membership.sum())
    owner = modal.App.lookup(settings.app_name, environment_name=settings.environment)
    image = modal.Image.from_id(worker_image_id)
    try:
        scores, seconds = _score_in_sandbox(owner, image, text, row["proposal_sha"], full,
                                           len(membership), k)
        row["runtime_s"] = round(seconds, 3)
        try:
            sub_scores, _ = _score_in_sandbox(owner, image, text, row["proposal_sha"],
                                             subset, len(subset_idx), k, subset=True)
        except api.Malformed as exc:
            if exc.reason in ("timeout", "sandbox_error"):
                raise
            raise api.Malformed("subset_dependent") from exc
        if not np.allclose(sub_scores, scores[subset_idx], rtol=1e-6, atol=1e-8):
            raise api.Malformed("subset_dependent")
        row["validation_auroc"] = round(auroc(scores[~membership], scores[membership]), 6)
    except api.Malformed as exc:
        row["status"] = exc.reason
    except Exception:
        row["status"] = "sandbox_error"
    return ledger.validate_row(row)


def _proposals(args) -> list[tuple[Path, str]]:
    if args.canaries:
        manifest = Path("proposals/canaries.json")
        rows = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or any(not isinstance(row, dict)
                                             or not isinstance(row.get("sha"), str)
                                             or re.fullmatch(r"[0-9a-f]{12}", row["sha"]) is None
                                             for row in rows):
            raise ValueError("Invalid canary manifest")
        return [(manifest.parent / f"{row['sha']}.py", "canary") for row in rows]
    if not args.proposal:
        raise ValueError("Choose --proposal FILE or --canaries")
    return [(Path(path), args.stream) for path in args.proposal]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--proposal", action="append", help="Hashed proposal file; repeatable")
    parser.add_argument("--canaries", action="store_true", help="Use proposals/canaries.json")
    parser.add_argument("--stream", default="unknown")
    parser.add_argument("--dataset", default="clinc150")
    parser.add_argument("--backbone", default="bert")
    parser.add_argument("--arm", default="ce")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ledger", default="runs/decisions/e3.jsonl")
    args = parser.parse_args(argv)
    spec = _spec(args.dataset, args.backbone, args.arm, args.seed)
    proposals = _proposals(args)
    if args.canaries and spec.arm != "ce":
        raise ValueError("The four canaries target the CE arm")
    candidates = [preflight(path, spec, stream) for path, stream in proposals]
    if args.command == "plan":
        print(json.dumps([row for _, row in candidates], indent=2, sort_keys=True))
        return 0
    existing = {(row["proposal_sha"], row["run"]): row for row in ledger.read_rows(args.ledger)}
    result = []

    def record(text: str, row: dict, worker_image_id: str | None = None) -> None:
        key = (row["proposal_sha"], row["run"])
        if key in existing:
            result.append(existing[key])
            return
        if row["status"] == "ready":
            if worker_image_id is None:
                raise RuntimeError("Worker image was not built")
            row = score_one.remote(text, spec.dataset, spec.backbone,
                                   spec.arm, spec.seed, worker_image_id, row["stream"])
        ledger.append_row(args.ledger, row)
        existing[key] = row
        result.append(row)

    remote_needed = any(row["status"] == "ready"
                        and (row["proposal_sha"], row["run"]) not in existing
                        for _, row in candidates)
    if remote_needed:
        with modal.enable_output():
            with e3_app.run(environment_name=settings.environment):
                worker_image_id = sandbox_image.object_id
                for text, row in candidates:
                    record(text, row, worker_image_id)
    else:
        for text, row in candidates:
            record(text, row)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
