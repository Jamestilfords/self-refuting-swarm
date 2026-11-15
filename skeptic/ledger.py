"""Append-only E3 validation ledger. Test data and raw scores are never rows."""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
from pathlib import Path

SCHEMA_VERSION = 1
FIELDS = frozenset({
    "schema_version", "proposal_sha", "run", "recipe_hash", "stream", "status",
    "validation_auroc", "runtime_s", "n_id", "n_oos",
})
STATUSES = frozenset({
    "ok", "syntax", "header", "import", "io", "name", "import_error",
    "runtime_error", "length", "nan", "constant", "probs",
    "subset_dependent", "timeout", "sandbox_error", "arm_mismatch",
})
_SHA = re.compile(r"[0-9a-f]{12}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_RUN = re.compile(r"(?:clinc150|banking77)_(?:bert|roberta)/(?:ce|edl)_seed[0-4]\Z")
_STREAM = re.compile(r"[A-Za-z0-9_-]{1,32}\Z")


def validate_row(row: dict, *, allow_ready: bool = False) -> dict:
    """Accept only the public validation summary schema, never arbitrary keys."""
    if not isinstance(row, dict) or set(row) != FIELDS:
        raise ValueError("Ledger row must have exactly the public E3 fields")
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported ledger schema version")
    for key, pattern in (("proposal_sha", _SHA), ("run", _RUN),
                         ("recipe_hash", _HASH), ("stream", _STREAM)):
        if not isinstance(row[key], str) or not pattern.fullmatch(row[key]):
            raise ValueError(f"Invalid ledger {key}")
    if row["status"] not in STATUSES and not (allow_ready and row["status"] == "ready"):
        raise ValueError("Invalid ledger status")
    for key in ("n_id", "n_oos"):
        value = row[key]
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"Invalid ledger {key}")
    runtime = row["runtime_s"]
    if runtime is not None and (type(runtime) not in (int, float)
                                or not math.isfinite(runtime) or runtime < 0):
        raise ValueError("Invalid ledger runtime_s")
    auroc = row["validation_auroc"]
    if row["status"] == "ok":
        if (type(auroc) not in (int, float) or not math.isfinite(auroc)
                or not 0 <= auroc <= 1 or row["n_id"] is None or row["n_oos"] is None):
            raise ValueError("Successful row needs a finite validation AUROC and split sizes")
    elif auroc is not None:
        raise ValueError("Failed row cannot have a validation AUROC")
    return row


def _rows_from_handle(handle) -> list[dict]:
    handle.seek(0)
    rows = []
    for number, line in enumerate(handle, 1):
        try:
            rows.append(validate_row(json.loads(line)))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid ledger row {number}: {exc}") from exc
    return rows


def read_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return _rows_from_handle(handle)


def append_row(path: str | Path, row: dict) -> bool:
    """Append once by (proposal, run). Identical retries are harmless."""
    row = validate_row(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        for old in _rows_from_handle(handle):
            if (old["proposal_sha"], old["run"]) == (row["proposal_sha"], row["run"]):
                if old != row:
                    raise ValueError("Conflicting row for proposal and run")
                return False
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        return True
