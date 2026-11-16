"""Open the sealed test scores after the published freeze is on main.

Writes ``runs/unblind.json`` and one ``opened`` line in ``runs/guard.log``.
A missing sealed file is listed and the command exits non-zero. No test
metric is printed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from skeptic.rules.common import REGISTERED_CANDIDATES_PER_CELL, REFERENCE_BASELINES
from skeptic.score_artifacts import CELLS
from skeptic.store import MANIFEST_RELATIVE, verify_published_freeze


def expected_files(manifest: dict) -> dict[str, list[str]]:
    """Relative paths under the sealed root, per cell. References are included."""
    out = {}
    for cell, group in CELLS.items():
        shas = list(manifest["proposals"][cell])
        shas.extend(manifest["references"][cell].values())
        unique = sorted(set(shas))
        out[cell] = [f"scores/{group}/{sha}.npz" for sha in unique]
    return out


def open_sealed(repo: Path, sealed_root: Path, *, require_registered: bool = True) -> dict:
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
    missing = []
    hashes = {}
    for paths in files.values():
        for relative in paths:
            path = sealed_root / relative
            if path.is_symlink() or not path.is_file():
                missing.append(relative)
                continue
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    plain = sealed_root / "plain_ece.json"
    if plain.is_symlink() or not plain.is_file():
        missing.append("plain_ece.json")
    else:
        hashes["plain_ece.json"] = hashlib.sha256(plain.read_bytes()).hexdigest()
    if missing:
        raise FileNotFoundError("sealed files missing:\n" + "\n".join(missing))
    record = {
        "schema_version": 1,
        "permit": {"commit": permit.commit, "digest": permit.digest},
        "counts": {cell: len(paths) for cell, paths in files.items()},
        "files": hashes,
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

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true",
                        help="Fixture use: do not require 492 files per cell")
    args = parser.parse_args(argv)
    try:
        record = open_sealed(args.repo, args.sealed, require_registered=not args.allow_partial)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"status": "opened", "counts": record["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
