"""Canonical ledger validation and matched proposal orderings."""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
from skeptic.score_artifacts import CELLS, ScoreArtifact, load_scores, auroc
PROPOSAL_STREAMS = frozenset({"templated", "llm"})
# Every sha in proposals/baselines.json. Replay retags these to stream baseline,
# including vacuity, then keeps them out of the orderings.
REGISTERED_SCORE_SHAS = frozenset({
    "b1e4ba2e46d5",  # energy@ce
    "39d3e8c0e810",  # energy@edl
    "a5325abe3321",  # msp@edl
    "3fee8950d4d1",  # d_port@ce
    "926c1ced9316",  # d_full@ce
    "fe47292dc266",  # msp@ce, also a templated scorecard file
    "6cc76a18b82b",  # vacuity@edl, a registered score and not a G2 baseline
    "71faf9e6269e",  # d_port@edl, also a templated scorecard file
    "2de1c4943af0",  # d_full@edl, also a templated scorecard file
})
# Couplings that still declare PARENTS. The files stay as generated.
# Replay and the scorecard ignore PARENTS for these shas, so G6 does not run.
IGNORE_PARENTS = frozenset({"03e0017097f5", "cddf9272a0a9", "2d5e3b3523bd"})
# Names the scorecard uses for two of those files. Pair 7 is 2d5e3b3523bd.
IGNORE_PARENT_NAMES = IGNORE_PARENTS | frozenset({
    "ecdf_add_log_d_full_edl",
    "qsigmoid_mul_d_port_edl",
})
# Published with the decision manifest. These hashes are scored on the sealed
# test set and are not candidates in any ordering.
REFERENCE_BASELINES = {
    "energy_ce": "b1e4ba2e46d5",
    "d_port_ce": "3fee8950d4d1",
    "d_full_ce": "926c1ced9316",
    "d_port_edl": "71faf9e6269e",
    "d_full_edl": "2de1c4943af0",
}
REFERENCE_ARMS = {
    "energy_ce": "ce",
    "d_port_ce": "ce",
    "d_full_ce": "ce",
    "d_port_edl": "edl",
    "d_full_edl": "edl",
}
FROZEN_RULES = (
    "R0", "R1", "R2",
    "R2-noG1", "R2-noG2", "R2-noG3", "R2-noG5", "R2-noG6",
)
ABLATION_SKIP = {
    "R2": frozenset(),
    "R2-noG1": frozenset({"G1"}),
    "R2-noG2": frozenset({"G2"}),
    "R2-noG3": frozenset({"G3"}),
    "R2-noG5": frozenset({"G5"}),
    "R2-noG6": frozenset({"G6"}),
}
ACCEPT_GATES = ("G1", "G2", "G3", "G5", "G6")
# In-scope length of the sealed test arrays. CLINC150 is 4500; Banking77-OOS is 2000.
TEST_N_IN = {
    "clinc150/bert": 4500,
    "clinc150/roberta": 4500,
    "banking77_oos/bert": 2000,
    "banking77_oos/roberta": 2000,
}
REGISTERED_CANDIDATES_PER_CELL = 487
_SHA = re.compile(r"[0-9a-f]{12}\Z")


def accept_gates_ok(rule: str, gates) -> bool:
    """An R2 accept records pass, or skipped, for each gate that rule removes."""
    if not str(rule).startswith("R2"):
        return True
    skip = ABLATION_SKIP.get(str(rule))
    if skip is None or not isinstance(gates, dict):
        return False
    return all(gates.get(gate) == ("skipped" if gate in skip else "pass") for gate in ACCEPT_GATES)


def reference_block(cells) -> dict[str, dict[str, str]]:
    return {cell: dict(REFERENCE_BASELINES) for cell in cells}

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
    """Registered NumPy permutations of scored proposals, seeded per cell."""
    if cell not in CELLS or type(count) is not int or count <= 0 or type(seed) is not int or seed < 0:
        raise ValueError("Cell, ordering count and seed must be registered")
    shas = sorted(sha for (sha, row_cell), row in rows.items()
                  if row_cell == cell and row["stream"] in PROPOSAL_STREAMS
                  and row["status"] == "ok")
    if not shas:
        raise ValueError("No proposals for cell")
    rng = np.random.default_rng(seed)
    return [tuple(rng.permutation(shas)) for _ in range(count)]
