"""Plain-model test ECE for the two arms, after the decision freeze is published.

The sealer stores scores and does not store a proposal's probabilities.
Final-pick ECE for a proposal that would have returned probabilities uses the
arm mean written here. The definition matches validation ``ece_plain``:
softmax ECE for CE, Dirichlet ECE for EDL, 15 bins, in-scope labels only.

``python -m modal_jobs.plain_ece`` verifies the freeze locally, then writes
``/sealed/plain_ece.json`` on the sealed volume. The return value is a status.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import modal
import numpy as np

from modal_jobs.app import sealed_volume, settings
from modal_jobs.config import SEALED_MOUNT
from modal_jobs.sandbox import runner_image

REPO = Path(__file__).resolve().parents[1]
ece_image = (
    runner_image
    .pip_install("torch==2.15.0", index_url="https://download.pytorch.org/whl/cpu")
    .env({**settings.container_env(), "SKEPTIC_ER1_SRC": "/opt/er1/src", "PYTHONPATH": "/opt/er1/src"})
)
ece_app = modal.App(f"{settings.app_name}-plain-ece")


def plain_model_ece(logits, labels, arm: str, bins: int = 15) -> float:
    """In-scope ECE of the untouched model. Same function the ledger stores.

    ``softmax_ece`` and ``dirichlet_ece`` use the ER-1 15-bin ECE. ``bins`` is
    accepted so a caller cannot silently switch definitions.
    """
    from skeptic.rules.critic_bridge import ensure_er1_path
    import torch

    opt = Path("/opt/er1/src")
    if opt.is_dir():
        os.environ.setdefault("SKEPTIC_ER1_SRC", str(opt))
    ensure_er1_path()
    from skeptic.critic.er1 import dirichlet_ece, logits_to_alpha, softmax_ece

    if bins != 15:
        raise ValueError("plain-model ECE uses the registered 15 bins")
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if arm == "ce":
        return float(softmax_ece(logits, labels))
    if arm == "edl":
        alpha = logits_to_alpha(torch.from_numpy(logits), 15.0, 1e-6)
        return float(dirichlet_ece(alpha.numpy(), labels))
    raise ValueError(f"arm must be ce or edl, got {arm!r}")


def write_plain_ece(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _existing_plain_ece(root: Path, commit: str, digest: str) -> str | None:
    from skeptic.store import GuardError

    destination = root / "plain_ece.json"
    if not destination.exists():
        return None
    if destination.is_symlink() or not destination.is_file():
        raise GuardError("plain_ece.json cannot be a symlink")
    try:
        existing = json.loads(destination.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GuardError("plain_ece.json provenance does not match the freeze") from exc
    if not isinstance(existing, dict):
        raise GuardError("plain_ece.json provenance does not match the freeze")
    if existing.get("decision_commit") == commit and existing.get("decision_digest") == digest:
        return "already_present"
    raise GuardError("plain_ece.json provenance does not match the freeze")


@ece_app.function(image=ece_image, volumes={SEALED_MOUNT: sealed_volume},
                 cpu=2, memory=8192, timeout=1800)
def compute_plain_ece(commit: str, digest: str) -> dict:
    """Write /sealed/plain_ece.json. Returns only status and counts."""
    from skeptic.rules.common import TEST_N_IN
    from skeptic.score_artifacts import CELLS
    from skeptic.store import GuardError, SealedStore

    sealed_volume.reload()
    root = Path(SEALED_MOUNT)
    store = SealedStore(root, REPO)
    store.unseal_remote(commit, digest)
    present = _existing_plain_ece(root, commit, digest)
    if present:
        return {"status": present}
    payload: dict = {"decision_commit": commit, "decision_digest": digest}
    count = 0
    for cell, group in CELLS.items():
        payload[cell] = {}
        for arm in ("ce", "edl"):
            values = []
            for seed in range(5):
                run = f"{group}/{arm}_seed{seed}"
                logits = np.asarray(store.read_array(run, "test", "logits"))
                labels = np.asarray(store.read_array(run, "test", "labels"))
                if len(logits) != TEST_N_IN[cell] or len(labels) != TEST_N_IN[cell]:
                    raise GuardError(f"In-scope test cache for {run} has the wrong length")
                value = plain_model_ece(logits, labels, arm)
                if not np.isfinite(value):
                    raise GuardError(f"plain-model ECE is not finite: {run}")
                values.append(float(value))
                count += 1
            payload[cell][arm] = values
    destination = root / "plain_ece.json"
    temporary = root / f".plain_ece.{uuid4().hex}.tmp"
    write_plain_ece(temporary, payload)
    try:
        os.replace(temporary, destination)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise GuardError("plain_ece.json could not be published") from exc
    sealed_volume.commit()
    return {"status": "written", "cells": len(CELLS), "values": count}


def main(argv=None) -> int:
    import argparse

    from skeptic.store import GuardError, verify_published_freeze

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    try:
        permit = verify_published_freeze(args.repo)
    except GuardError as exc:
        print(json.dumps({"status": "freeze_required", "error": str(exc)}))
        return 1
    try:
        with modal.enable_output():
            with ece_app.run(environment_name=settings.environment):
                result = compute_plain_ece.remote(permit.commit, permit.digest)
    except GuardError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") in {"written", "already_present"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
