"""Canonical ledger validation and matched proposal orderings."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from skeptic.score_artifacts import CELLS, ScoreArtifact, load_scores, auroc
PROPOSAL_STREAMS = frozenset({"templated", "llm"})
_SHA = re.compile(r"[0-9a-f]{12}\Z")

def load_ledger(path: str | Path) -> dict[tuple[str, str], dict]:
    """Read the richer, canonical runs/ledger.jsonl format once."""
    rows = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                sha, cell = row["sha"], row["cell"]
                if not _SHA.fullmatch(sha) or cell not in CELLS or row["stream"] not in \
                        (PROPOSAL_STREAMS | {"baseline", "canary"}):
                    raise ValueError("Invalid proposal identity")
                key = (sha, cell)
                if key in rows:
                    raise ValueError("Duplicate proposal/cell row")
                val = row.get("val")
                if row.get("status") == "ok":
                    aurocs = val.get("auroc") if isinstance(val, dict) else None
                    if not isinstance(aurocs, list) or len(aurocs) != 5 \
                            or any(type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1
                                   for v in aurocs):
                        raise ValueError("Successful row needs five validation AUROCs")
                rows[key] = row
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid canonical ledger row {number}: {exc}") from exc
    if not rows:
        raise ValueError("Canonical ledger is empty")
    return rows

def matched_orderings(rows: dict[tuple[str, str], dict], cell: str,
                      count: int, seed: int) -> list[tuple[str, ...]]:
    """Stable hash ordering; no Python or NumPy RNG-version dependence."""
    if cell not in CELLS or type(count) is not int or count <= 0 or type(seed) is not int or seed < 0:
        raise ValueError("Cell, ordering count and seed must be registered")
    shas = sorted(sha for (sha, row_cell), row in rows.items()
                  if row_cell == cell and row["stream"] in PROPOSAL_STREAMS)
    if not shas:
        raise ValueError("No proposals for cell")
    result = []
    for ordering in range(count):
        prefix = f"{seed}:{cell}:{ordering}:".encode()
        result.append(tuple(sorted(shas, key=lambda sha: hashlib.sha256(prefix + sha.encode()).digest())))
    return result
