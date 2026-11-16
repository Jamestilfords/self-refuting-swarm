"""Hash sealed score files after the published freeze. Returns digests only."""

from __future__ import annotations

import re
from pathlib import Path

import modal
import numpy as np

from modal_jobs.app import sealed_volume, settings
from modal_jobs.config import SEALED_MOUNT
from modal_jobs.sandbox import runner_image
from skeptic.analysis.sources import file_sha256
from skeptic.store import GuardError, verify_remote_freeze

unblind_app = modal.App(f"{settings.app_name}-unblind")
_GROUPS = "clinc150_bert|clinc150_roberta|banking77_bert|banking77_roberta"
_SCORE = re.compile(rf"scores/(?:{_GROUPS})/[0-9a-f]{{12}}\.npz\Z")


def _require_expected_path(relative: str) -> None:
    if not isinstance(relative, str) or not relative or relative.startswith(("/", "\\")) or "\\" in relative:
        raise GuardError("Sealed path is not relative")
    if ".." in Path(relative).parts:
        raise GuardError("Sealed path is not relative")
    if relative != "plain_ece.json" and _SCORE.fullmatch(relative) is None:
        raise GuardError("Sealed path is not an expected file")


def _inside_root(root: Path, path: Path) -> bool:
    current = path
    while True:
        if current.is_symlink():
            raise GuardError("Sealed path is a symlink")
        if current == root:
            return True
        if root not in current.parents:
            return False
        current = current.parent


def _provenance_matches(path: Path, commit: str, digest: str) -> bool:
    try:
        with np.load(path, allow_pickle=False) as saved:
            return (str(saved["decision_commit"]) == commit
                    and str(saved["decision_digest"]) == digest
                    and str(saved["proposal_sha"]) == path.stem)
    except (OSError, ValueError, KeyError):
        return False


def hash_sealed_files(root: Path, commit: str, digest: str, expected: list[str]) -> dict:
    """sha256 of each expected file under ``root``. Provenance mismatches are missing."""
    files: dict[str, str] = {}
    missing: list[str] = []
    for relative in expected:
        _require_expected_path(relative)
        path = root / relative
        if not path.exists() or not path.is_file():
            missing.append(relative)
            continue
        _inside_root(root, path)
        if relative != "plain_ece.json" and not _provenance_matches(path, commit, digest):
            missing.append(relative)
            continue
        files[relative] = file_sha256(path)
    return {"files": files, "missing": missing}


@unblind_app.function(image=runner_image,
                      volumes={SEALED_MOUNT: sealed_volume.with_mount_options(read_only=True)},
                      cpu=2, memory=4096, timeout=1800)
def hash_sealed(commit: str, digest: str, expected: list[str]) -> dict:
    """sha256 of each expected file under /sealed. Returns {"files": {rel: hex}, "missing": [rel]}."""
    sealed_volume.reload()
    verify_remote_freeze(commit, digest)
    return hash_sealed_files(Path(SEALED_MOUNT), commit, digest, expected)
