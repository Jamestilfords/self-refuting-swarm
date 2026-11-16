"""Plain-model test ECE for the two arms, after the decision freeze is published.

The sealer stores scores and does not store a proposal's probabilities.
Final-pick ECE for a proposal that would have returned probabilities uses the
arm mean written here. The definition matches validation ``ece_plain``:
softmax ECE for CE, Dirichlet ECE for EDL, 15 bins, in-scope labels only.

The local command reads a mounted sealed root. ``--remote`` runs the same
computation on the ``skeptic-sealed`` volume and leaves the file there.
Neither path prints the ECE values.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import modal
import numpy as np

from modal_jobs.app import sealed_volume, settings
from modal_jobs.config import SEALED_MOUNT
from modal_jobs.sandbox import runner_image

# Directory group, analysis cell, class count. Banking77-OOS omits ``_oos``
# in the sealed directory and keeps it in the cell name.
_GROUPS = (
    ("clinc150_bert", "clinc150/bert", 150),
    ("clinc150_roberta", "clinc150/roberta", 150),
    ("banking77_bert", "banking77_oos/bert", 50),
    ("banking77_roberta", "banking77_oos/roberta", 50),
)
_RUNS = tuple(
    (group, cell, n_classes, arm, seed)
    for group, cell, n_classes in _GROUPS
    for arm in ("ce", "edl")
    for seed in range(5)
)

ece_image = runner_image.pip_install("torch==2.15.0")
ece_app = modal.App(f"{settings.app_name}-ece")


def plain_model_ece(logits, labels, arm: str, bins: int = 15) -> float:
    """In-scope ECE of the untouched model. Same function the ledger stores.

    ``softmax_ece`` and ``dirichlet_ece`` use the ER-1 15-bin ECE. ``bins`` is
    accepted so a caller cannot silently switch definitions.
    """
    from skeptic.critic.er1 import dirichlet_ece, logits_to_alpha, softmax_ece
    import torch

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


def collect_plain_ece(store, permit) -> dict:
    """ECE of each sealed in-scope test cache. One float per seed, five per arm."""
    from skeptic.rules.common import TEST_N_IN
    from skeptic.store import GuardError

    payload = {
        "decision_commit": permit.commit,
        "decision_digest": permit.digest,
    }
    for group, cell, n_classes, arm, seed in _RUNS:
        run = f"{group}/{arm}_seed{seed}"
        logits = np.asarray(store.read_array(run, "test", "logits"))
        labels = np.asarray(store.read_array(run, "test", "labels"))
        n_in = TEST_N_IN[cell]
        if logits.shape != (n_in, n_classes) or labels.shape != (n_in,):
            raise GuardError(f"In-scope test cache for {run} has the wrong shape")
        if not np.isfinite(logits).all():
            raise GuardError(f"In-scope test logits are non-finite: {run}")
        if not np.issubdtype(labels.dtype, np.integer) \
                or (labels < 0).any() or (labels >= n_classes).any():
            raise GuardError(f"In-scope test labels are invalid: {run}")
        payload.setdefault(cell, {})
        payload[cell].setdefault(arm, []).append(plain_model_ece(logits, labels, arm))
    return payload


def write_plain_ece(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def existing_plain_ece(root: Path, permit) -> str | None:
    """Return ``already_present`` when this freeze already wrote the file."""
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
    if existing.get("decision_commit") == permit.commit \
            and existing.get("decision_digest") == permit.digest:
        return "already_present"
    raise GuardError("plain_ece.json provenance does not match the freeze")


def publish_plain_ece(root: Path, payload: dict) -> str:
    """Write ``plain_ece.json`` for this freeze. The caller has already checked the file is absent."""
    from skeptic.store import GuardError, UnsealPermit

    permit = UnsealPermit(payload["decision_digest"], payload["decision_commit"])
    if existing_plain_ece(root, permit) == "already_present":
        return "already_present"
    destination = root / "plain_ece.json"
    temporary = root / f".plain_ece.{uuid4().hex}.tmp"
    write_plain_ece(temporary, payload)
    try:
        temporary.replace(destination)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise GuardError("plain_ece.json could not be published") from exc
    return "written"


def _write_local(sealed: Path, repo: Path, permit) -> str:
    from skeptic.store import SealedStore

    present = existing_plain_ece(sealed, permit)
    if present:
        return present
    store = SealedStore(sealed, repo)
    store._permit = permit
    return publish_plain_ece(sealed, collect_plain_ece(store, permit))


@ece_app.function(image=ece_image, volumes={SEALED_MOUNT: sealed_volume},
                 cpu=2, memory=8192, timeout=1800, max_containers=1)
def write_plain_ece_remote(commit: str, digest: str) -> dict:
    """Read in-scope test logits on skeptic-sealed and store plain_ece.json there."""
    from skeptic.store import GuardError, SealedStore

    sealed_volume.reload()
    root = Path(SEALED_MOUNT)
    store = SealedStore(root, root)
    try:
        permit = store.unseal_remote(commit, digest)
        present = existing_plain_ece(root, permit)
        status = present or publish_plain_ece(root, collect_plain_ece(store, permit))
    except GuardError as exc:
        return {"status": "needs_sealed_logits", "error": str(exc)}
    if status == "written":
        sealed_volume.commit()
    return {"status": status, "path": str(root / "plain_ece.json")}


def _launch_remote(permit) -> int:
    with modal.enable_output():
        with ece_app.run(environment_name=settings.environment):
            result = write_plain_ece_remote.remote(permit.commit, permit.digest)
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") in {"written", "already_present"} else 2


def main(argv=None) -> int:
    import argparse

    from skeptic.store import GuardError, verify_published_freeze

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--sealed", type=Path, required=True,
                        help="Mounted skeptic-sealed root that already holds test logits")
    parser.add_argument("--remote", action="store_true",
                        help="Run on the skeptic-sealed Modal volume and write /sealed/plain_ece.json")
    args = parser.parse_args(argv)
    try:
        permit = verify_published_freeze(args.repo)
    except GuardError as exc:
        print(json.dumps({"status": "freeze_required", "error": str(exc)}))
        return 1
    if args.remote:
        return _launch_remote(permit)
    try:
        status = _write_local(args.sealed, args.repo, permit)
    except GuardError as exc:
        print(json.dumps({
            "status": "needs_sealed_logits",
            "error": str(exc),
            "decision_commit": permit.commit,
            "decision_digest": permit.digest,
            "path": str(args.sealed / "plain_ece.json"),
        }))
        return 2
    print(json.dumps({"status": status, "path": str(args.sealed / "plain_ece.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
