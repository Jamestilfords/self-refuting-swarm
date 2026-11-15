"""E4 sealed-store read guard and published decision-log verification.

This module is for trusted jobs. Untrusted proposal code must never run in a
process with the sealed volume mounted; a Python guard cannot restrict code
that already has direct filesystem access to that mount.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HASH_RELATIVE = Path("runs/DECISIONS.sha256")
DECISIONS_RELATIVE = Path("runs/decisions")
MANIFEST_RELATIVE = Path("runs/decision_manifest.json")
REGISTRATION_RELATIVE = Path("PREREGISTRATION.md")
PUBLISHED_REPO = "Jamestilfords/self-refuting-swarm"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SHA12 = re.compile(r"[0-9a-f]{12}\Z")
_SPLITS = frozenset({"train", "val", "oos_val", "test", "oos_test"})
_FIELDS = frozenset({"logits", "feats", "pooled", "labels"})
_TEST = frozenset({"test", "oos_test"})


class GuardError(RuntimeError):
    """The requested sealed read or unblind operation is not authorized."""


def _r2_gates_pass(gates: object) -> bool:
    if not isinstance(gates, dict):
        return False
    return all(gates.get(gate) == "pass" for gate in ("G1", "G2", "G3", "G5", "G6"))


def _decision_files(repo: Path) -> list[Path]:
    root = repo / DECISIONS_RELATIVE
    if not root.is_dir() or root.is_symlink():
        raise GuardError("Decision directory is missing")
    # E3 writes its own operational log here. Only the three E5 rule folders
    # constitute the frozen decision set.
    rule_dirs = [root / rule for rule in ("R0", "R1", "R2")]
    if any(not path.is_dir() or path.is_symlink() for path in rule_dirs):
        raise GuardError("R0, R1 and R2 decision directories are required")
    files = sorted((p for directory in rule_dirs for p in directory.rglob("*.jsonl")),
                   key=lambda p: p.relative_to(root).as_posix())
    if not files or any(p.is_symlink() or not p.is_file() for p in files):
        raise GuardError("Decision logs are missing or contain a symlink")
    other = [p for directory in rule_dirs for p in directory.rglob("*")
             if p.is_symlink() or (p.is_file() and p.suffix != ".jsonl")]
    if other:
        raise GuardError("Unexpected file in decision directory")
    return files


def _manifest_paths(repo: Path) -> tuple[set[str], dict[str, set[str]]]:
    path = repo / MANIFEST_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise GuardError("Decision manifest is missing")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise GuardError("Invalid decision manifest") from exc
    paths = manifest.get("files") if isinstance(manifest, dict) else None
    proposals = manifest.get("proposals") if isinstance(manifest, dict) else None
    cells = ["banking77_oos/bert", "banking77_oos/roberta",
             "clinc150/bert", "clinc150/roberta"]
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("rules") != ["R0", "R1", "R2"]
            or manifest.get("cells") != cells
            or type(manifest.get("orderings")) is not int or manifest["orderings"] <= 0
            or not isinstance(paths, list) or not paths
            or any(not isinstance(p, str) or not re.fullmatch(
                r"runs/decisions/R[012]/ord_[0-9]{3,}\.jsonl", p) for p in paths)
            or len(paths) != len(set(paths))
            or not isinstance(proposals, dict) or set(proposals) != set(cells)
            or any(not isinstance(shas, list) or not shas
                   or any(not isinstance(sha, str) or not _SHA12.fullmatch(sha)
                          for sha in shas)
                   or shas != sorted(set(shas))
                   for shas in proposals.values())):
        raise GuardError("Invalid decision manifest file list")
    expected = {f"runs/decisions/{rule}/ord_{i:03d}.jsonl"
                for rule in manifest["rules"] for i in range(manifest["orderings"])}
    if set(paths) != expected:
        raise GuardError("Decision manifest does not cover all rules and orderings")
    return set(paths), {cell: set(shas) for cell, shas in proposals.items()}


def decision_digest(repo_root: str | Path) -> str:
    """Hash exact relative paths and bytes for every declared decision log."""
    repo = Path(repo_root)
    files = _decision_files(repo)
    expected, proposals = _manifest_paths(repo)
    actual = {p.relative_to(repo).as_posix() for p in files}
    if actual != expected:
        raise GuardError("Decision files do not match the manifest")
    digest = hashlib.sha256(b"skeptic-decisions-v1\0")
    sequences: dict[tuple[int, str], tuple[str, ...]] = {}
    baselines: dict[str, str] = {}
    for path in files:
        relative = path.relative_to(repo).as_posix().encode("utf-8")
        data = path.read_bytes()
        if not data or not data.endswith(b"\n"):
            raise GuardError(f"Incomplete decision log: {path.name}")
        rule = path.parent.name
        ordering = int(path.stem.removeprefix("ord_"))
        by_cell: dict[str, dict[int, dict]] = {}
        for number, line in enumerate(data.splitlines(), 1):
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError) as exc:
                raise GuardError(f"Invalid decision JSON in {path.name}:{number}") from exc
            if not isinstance(row, dict) or row.get("schema_version") != 1 \
                    or row.get("rule") != rule or row.get("ordering") != ordering \
                    or row.get("cell") not in ("clinc150/bert", "clinc150/roberta",
                                               "banking77_oos/bert", "banking77_oos/roberta") \
                    or type(row.get("idx")) is not int or row["idx"] < 0 \
                    or not isinstance(row.get("sha"), str) or not _SHA12.fullmatch(row["sha"]) \
                    or not isinstance(row.get("incumbent"), str) \
                    or not _SHA12.fullmatch(row["incumbent"]) \
                    or row.get("verdict") not in ("ACCEPT", "REJECT", "INERT", "SCORE-MATCHED",
                                                  "NO EFFECT", "UNDERPOWERED", "MISCALIBRATES", "BUG") \
                    or (rule in ("R0", "R1") and row["verdict"] not in ("ACCEPT", "REJECT")) \
                    or (rule == "R2" and row["verdict"] == "ACCEPT" and
                        not _r2_gates_pass(row.get("gates"))) \
                    or any("test" in key.lower() or "truth" in key.lower() for key in row):
                raise GuardError(f"Invalid decision row in {path.name}:{number}")
            cell_rows = by_cell.setdefault(row["cell"], {})
            if row["idx"] in cell_rows:
                raise GuardError(f"Duplicate position in {path.name}")
            cell_rows[row["idx"]] = row
        if set(by_cell) != {"clinc150/bert", "clinc150/roberta",
                            "banking77_oos/bert", "banking77_oos/roberta"}:
            raise GuardError(f"Missing cell in {path.name}")
        for cell, rows in by_cell.items():
            if sorted(rows) != list(range(len(rows))):
                raise GuardError(f"Missing position in {path.name}")
            sequence = tuple(rows[index]["sha"] for index in range(len(rows)))
            if len(sequence) != len(set(sequence)) or set(sequence) != proposals[cell]:
                raise GuardError(f"Proposal set differs from manifest for {cell}")
            key = (ordering, cell)
            incumbent = rows[0]["incumbent"]
            if cell in baselines and baselines[cell] != incumbent:
                raise GuardError(f"Rules start from different incumbents for {cell}")
            baselines[cell] = incumbent
            for index in range(len(rows)):
                row = rows[index]
                if row["incumbent"] != incumbent:
                    raise GuardError(f"Broken incumbent chain in {path.name} for {cell}")
                if row["verdict"] == "ACCEPT":
                    incumbent = row["sha"]
            if key in sequences and sequences[key] != sequence:
                raise GuardError(f"Rules have different proposal orderings for {cell}")
            sequences[key] = sequence
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    manifest_data = (repo / MANIFEST_RELATIVE).read_bytes()
    digest.update(b"manifest\0" + len(manifest_data).to_bytes(8, "big") + manifest_data)
    return digest.hexdigest()


def _check_registration(repo: Path) -> None:
    registration = repo / REGISTRATION_RELATIVE
    if not registration.is_file() or registration.is_symlink():
        raise GuardError("Preregistration is missing")
    registration_text = registration.read_text(encoding="utf-8")
    if not re.search(r"(?mi)^\*{0,2}Status:\s*FROZEN\b", registration_text) \
            or "TODO:" in registration_text:
        raise GuardError("Preregistration is still a draft or lacks FROZEN status")


def write_decision_hash(repo_root: str | Path) -> str:
    """Create a hash once, after the decision manifest and all logs exist."""
    repo = Path(repo_root)
    _check_registration(repo)
    value = decision_digest(repo)
    target = repo / HASH_RELATIVE
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="ascii") as handle:
            handle.write(value + "\n")
    except FileExistsError as exc:
        raise GuardError("Decision hash already exists; do not replace a freeze") from exc
    return value


def verify_decision_hash(repo_root: str | Path) -> str:
    repo = Path(repo_root)
    _check_registration(repo)
    path = repo / HASH_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise GuardError("Decision hash is missing")
    try:
        text = path.read_text(encoding="ascii")
    except UnicodeError as exc:
        raise GuardError("Invalid decision hash file") from exc
    if not text.endswith("\n") or not _HEX64.fullmatch(text[:-1]):
        raise GuardError("Invalid decision hash file")
    if text[:-1] != decision_digest(repo):
        raise GuardError("Decision hash differs from the logs")
    return text[:-1]


def _git_bytes(repo: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(["git", *args], cwd=repo, capture_output=True,
                                check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GuardError(f"Git proof failed: {' '.join(args[:2])}") from exc
    if result.returncode:
        raise GuardError(f"Git proof failed: {' '.join(args[:2])}")
    return result.stdout


def _git(repo: Path, *args: str) -> str:
    return _git_bytes(repo, *args).decode("utf-8").strip()


@dataclass(frozen=True)
class UnsealPermit:
    digest: str
    commit: str


def verify_published_freeze(repo_root: str | Path) -> UnsealPermit:
    """Fail closed unless the exact local tree is clean and on published main."""
    repo = Path(repo_root)
    digest = verify_decision_hash(repo)
    if _git(repo, "status", "--porcelain"):
        raise GuardError("Working tree is dirty; unblinding requires a clean freeze checkout")
    origin = _git(repo, "remote", "get-url", "origin")
    if origin not in (f"https://github.com/{PUBLISHED_REPO}.git",
                      f"https://github.com/{PUBLISHED_REPO}",
                      f"git@github.com:{PUBLISHED_REPO}.git"):
        raise GuardError("Origin is not the registered GitHub repository")
    head = _git(repo, "rev-parse", "HEAD")
    remote_line = _git(repo, "ls-remote", "origin", "refs/heads/main")
    remote_head = remote_line.split()[0] if remote_line else ""
    if head != remote_head:
        raise GuardError("Decision hash commit is not the published main HEAD")
    # git status can omit ignored or assume-unchanged files. Verify every
    # frozen input against the published Git object, including its exact bytes.
    frozen = [repo / HASH_RELATIVE, repo / MANIFEST_RELATIVE,
              repo / REGISTRATION_RELATIVE, *_decision_files(repo)]
    for path in frozen:
        relative = path.relative_to(repo).as_posix()
        committed = _git_bytes(repo, "show", f"{head}:{relative}")
        if committed != path.read_bytes():
            raise GuardError(f"Published freeze differs from local file: {relative}")
    return UnsealPermit(digest, head)


class SealedStore:
    """Read only named arrays; test splits require a verified freeze permit."""

    def __init__(self, root: str | Path, repo_root: str | Path):
        self._root = Path(root)
        self._repo = Path(repo_root)
        self._permit: UnsealPermit | None = None

    def unseal(self) -> UnsealPermit:
        self._permit = None
        self._permit = verify_published_freeze(self._repo)
        return self._permit


    def guard_read(self, split: str) -> None:
        if split not in _SPLITS:
            raise GuardError("Unknown data split")
        if split in _TEST and self._permit is None:
            raise GuardError("Test split is sealed until the published decision hash is verified")

    def read_array(self, run: str, split: str, field: str) -> np.ndarray:
        self.guard_read(split)
        if field not in _FIELDS:
            raise GuardError("Unknown cache field")
        if run not in {f"{group}/{arm}_seed{seed}"
                       for group in ("clinc150_bert", "clinc150_roberta", "banking77_bert", "banking77_roberta")
                       for arm in ("ce", "edl") for seed in range(5)}:
            raise GuardError("Unknown E2 run")
        filename = "pooled.npz" if field == "pooled" else "cache.npz"
        path = self._root / run / filename
        group, seed_run = run.split("/")
        if any(p.is_symlink() for p in (self._root / group, self._root / group / seed_run, path)) \
                or not path.is_file():
            raise GuardError("Sealed cache is missing")
        key = f"{split}__{field}"
        with np.load(path, allow_pickle=False) as cache:
            if key not in cache.files:
                raise GuardError("Sealed cache array is missing")
            return np.asarray(cache[key])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E4 decision freeze and sealed-read guard")
    parser.add_argument("command", choices=("digest", "freeze", "verify", "unseal-check"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    if args.command == "digest":
        print(decision_digest(args.repo))
    elif args.command == "freeze":
        print(write_decision_hash(args.repo))
    elif args.command == "verify":
        print(verify_decision_hash(args.repo))
    else:
        permit = verify_published_freeze(args.repo)
        print(json.dumps({"commit": permit.commit, "decision_digest": permit.digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
