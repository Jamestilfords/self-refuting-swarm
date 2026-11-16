"""Open the sealed test scores after the published freeze is on main.

The local half verifies the freeze and writes ``runs/unblind.json``. File
bytes are hashed on the sealed volume; only digests and a missing-file list
come back. A missing sealed file is listed and the command exits non-zero.
No test metric is printed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from skeptic.rules.common import REGISTERED_CANDIDATES_PER_CELL, REFERENCE_BASELINES
from skeptic.score_artifacts import CELLS
from skeptic.store import MANIFEST_RELATIVE, GuardError, verify_published_freeze


def expected_files(manifest: dict) -> dict[str, list[str]]:
    """Relative paths under the sealed root, per cell. References are included."""
    out = {}
    for cell, group in CELLS.items():
        shas = list(manifest["proposals"][cell])
        shas.extend(manifest["references"][cell].values())
        unique = sorted(set(shas))
        out[cell] = [f"scores/{group}/{sha}.npz" for sha in unique]
    return out


def expected_remote_paths(manifest: dict) -> list[str]:
    """Score files plus ``plain_ece.json``, the list sent to the volume hasher."""
    paths = sorted({path for cell_paths in expected_files(manifest).values() for path in cell_paths})
    paths.append("plain_ece.json")
    return paths


def open_sealed(repo: Path, *, hasher, require_registered: bool = True) -> dict:
    permit = verify_published_freeze(repo)
    manifest = json.loads((repo / MANIFEST_RELATIVE).read_text(encoding="utf-8"))
    files = expected_files(manifest)
    if require_registered:
        for cell, paths in files.items():
            if len(paths) != REGISTERED_CANDIDATES_PER_CELL + len(REFERENCE_BASELINES):
                raise ValueError(
                    f"{cell} lists {len(paths)} sealed files; "
                    f"the registered set is {REGISTERED_CANDIDATES_PER_CELL + len(REFERENCE_BASELINES)}"
                )
    expected = expected_remote_paths(manifest)
    result = hasher(permit.commit, permit.digest, expected)
    hashes = dict(result.get("files") or {})
    missing = list(result.get("missing") or [])
    for relative in expected:
        if relative not in hashes and relative not in missing:
            missing.append(relative)
    if missing:
        raise FileNotFoundError("sealed files missing:\n" + "\n".join(missing))
    record = {
        "schema_version": 1,
        "permit": {"commit": permit.commit, "digest": permit.digest},
        "counts": {cell: len(paths) for cell, paths in files.items()},
        "files": {relative: hashes[relative] for relative in expected},
    }
    destination = repo / "runs" / "unblind.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    guard = repo / "runs" / "guard.log"
    stamp = datetime.now(timezone.utc).isoformat()
    with guard.open("a", encoding="utf-8") as handle:
        handle.write(
            f'{stamp}, purpose="unblind", commit="{permit.commit}", digest="{permit.digest}"\n'
        )
    return record


def main(argv=None) -> int:
    import argparse
    import sys

    import modal

    from modal_jobs.app import settings
    from modal_jobs.unblind_remote import hash_sealed, unblind_app

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--allow-partial", action="store_true",
                        help="Fixture use: do not require 492 files per cell")
    args = parser.parse_args(argv)

    def hasher(commit: str, digest: str, expected: list[str]) -> dict:
        with modal.enable_output():
            with unblind_app.run(environment_name=settings.environment):
                return hash_sealed.remote(commit, digest, expected)

    try:
        record = open_sealed(args.repo, hasher=hasher, require_registered=not args.allow_partial)
    except (FileNotFoundError, ValueError, OSError, GuardError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"status": "opened", "counts": record["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
