"""Load the ledger and the matched decision logs.

The loader checks three things before any test number is computed: each
ordering is a complete pass over the status-ok, non-canary stream; every rule
saw that stream in the same order; and each accepted row names the incumbent
left by the previous acceptance.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from skeptic.analysis.config import DEFAULT, AnalysisConfig
from skeptic.rules.common import REFERENCE_BASELINES, REGISTERED_SCORE_SHAS

_ORD = re.compile(r"ord_(\d+)$")
_REFERENCE = "energy_ce"
_REFERENCE_NAMES = {sha: name for name, sha in REFERENCE_BASELINES.items()}
_BINARY_VERDICTS = {"ACCEPT": True, "REJECT": False}


def canary_mask(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    if frame.empty:
        return mask
    if "canary" in frame.columns:
        mask = mask | frame["canary"].fillna(False).astype(bool)
    if "family" in frame.columns:
        mask = mask | frame["family"].astype(str).str.lower().eq("canary")
    if "stream" in frame.columns:
        mask = mask | frame["stream"].astype(str).str.lower().eq("canary")
    return mask


def eligible_shas(ledger: pd.DataFrame, cell: str | None = None) -> list[str]:
    """Registered candidates: status-ok, not a canary, and not a registered score."""
    frame = ledger
    if cell is not None and "cell" in frame.columns:
        frame = frame.loc[frame["cell"].astype(str) == str(cell)]
    if "status" in frame.columns:
        frame = frame.loc[frame["status"].astype(str) == "ok"]
    if frame.empty:
        return []
    frame = frame.loc[~canary_mask(frame)]
    if "stream" in frame.columns:
        frame = frame.loc[frame["stream"].astype(str).str.lower() != "baseline"]
    frame = frame.loc[~frame["sha"].astype(str).isin(REGISTERED_SCORE_SHAS)]
    seen: set[str] = set()
    ordered: list[str] = []
    for sha in frame["sha"].astype(str):
        if sha not in seen:
            seen.add(sha)
            ordered.append(sha)
    return ordered


def load_ledger(path: str | Path = "runs/ledger.jsonl") -> pd.DataFrame:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame(columns=["sha", "stream", "family", "arm", "cell", "status", "parents"])
    return pd.DataFrame(rows)


def decisions_tree_sha256(root: str | Path) -> str:
    """Content hash of a decision directory, used by the unseal guard."""
    directory = Path(root)
    digest = hashlib.sha256()
    if not directory.exists():
        digest.update(b"MISSING")
        return digest.hexdigest()
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    for path in files:
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _parse_payload(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("rows", "decisions"):
            if key in payload:
                return list(payload[key])
    raise ValueError(f"{path} is not a decision list")


def _iter_files(root: Path):
    if not root.is_dir():
        raise FileNotFoundError(f"decision directory {root} does not exist")
    for rule_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for path in sorted(rule_dir.rglob("*")):
            if not path.is_file() or path.suffix not in {".jsonl", ".json"}:
                continue
            match = _ORD.search(path.stem)
            if match is None:
                continue
            yield rule_dir.name, int(match.group(1)), path


def _cell_from_path(root: Path, rule: str, path: Path) -> str | None:
    relative = path.parent.relative_to(root / rule)
    if relative.parts == ():
        return None
    return "/".join(relative.parts)


def _coerce_accepted(rule: str, row: dict) -> bool:
    verdict = row.get("verdict")
    if "accepted" in row and row["accepted"] is not None:
        accepted = bool(row["accepted"])
        if verdict in _BINARY_VERDICTS and _BINARY_VERDICTS[verdict] != accepted:
            raise ValueError(
                f"{rule} decision for sha={row.get('sha')} has accepted {accepted} "
                f"and verdict {verdict}"
            )
        return accepted
    if str(rule).startswith("R2"):
        return verdict == "ACCEPT"
    if verdict in _BINARY_VERDICTS:
        return _BINARY_VERDICTS[verdict]
    raise ValueError(
        f"{rule} decision for sha={row.get('sha')} has neither accepted nor an ACCEPT/REJECT verdict"
    )


def load_decisions(
    root: str | Path = "runs/decisions",
    ledger: pd.DataFrame | None = None,
    *,
    config: AnalysisConfig = DEFAULT,
    reference: str = _REFERENCE,
) -> pd.DataFrame:
    """One row per decision. ``rule`` is the canonical name; ``ordering`` comes from ``ord_NNN``."""
    directory = Path(root)
    if ledger is None:
        ledger_path = directory.parent / "ledger.jsonl"
        if ledger_path.is_file():
            ledger = load_ledger(ledger_path)
    rows: list[dict] = []
    for directory_name, ordering, path in _iter_files(directory):
        rule = config.canonical_rule(directory_name)
        path_cell = _cell_from_path(directory, directory_name, path)
        for raw in _parse_payload(path):
            row = {key: value for key, value in raw.items() if key != "detail"}
            if "sha" not in row or "idx" not in row or "incumbent" not in row:
                raise ValueError(f"{path} is missing sha, idx, or incumbent")
            cell = str(row["cell"]) if row.get("cell") is not None else path_cell
            if cell is None:
                raise ValueError(f"{path} has no cell")
            if path_cell is not None and str(row.get("cell", path_cell)) != path_cell:
                raise ValueError(f"{path} cell does not match its directory")
            if "rule" in row and config.canonical_rule(str(row["rule"])) not in {rule, directory_name}:
                raise ValueError(f"{path} rule does not match its directory")
            if "ordering" in row and int(row["ordering"]) != ordering:
                raise ValueError(f"{path} ordering does not match its filename")
            row["rule"] = rule
            row["rule_dir"] = directory_name
            row["ordering"] = ordering
            row["cell"] = cell
            row["sha"] = str(row["sha"])
            row["idx"] = int(row["idx"])
            row["incumbent"] = _REFERENCE_NAMES.get(str(row["incumbent"]), str(row["incumbent"]))
            if row.get("comparator") is not None:
                row["comparator"] = _REFERENCE_NAMES.get(str(row["comparator"]), str(row["comparator"]))
            row["accepted"] = _coerce_accepted(rule, raw)
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"no decision rows under {directory}")
    _validate(frame, ledger, reference)
    return frame.reset_index(drop=True)


def _non_canary(frame: pd.DataFrame, ledger: pd.DataFrame | None) -> pd.DataFrame:
    mask = pd.Series(False, index=frame.index)
    if "canary" in frame.columns:
        mask = mask | frame["canary"].fillna(False).astype(bool)
    if ledger is not None and not ledger.empty:
        canaries = set(ledger.loc[canary_mask(ledger), "sha"].astype(str))
        mask = mask | frame["sha"].astype(str).isin(canaries)
    return frame.loc[~mask].copy()


def _validate(frame: pd.DataFrame, ledger: pd.DataFrame | None, reference: str) -> None:
    body = _non_canary(frame, ledger)
    sequences: dict[tuple[str, int], dict[str, tuple[str, ...]]] = {}
    for (rule, cell, ordering), group in body.groupby(["rule", "cell", "ordering"], sort=True):
        ordered = group.sort_values(["idx", "sha"])
        idx = [int(value) for value in ordered["idx"].tolist()]
        if len(idx) != len(set(idx)) or idx != list(range(len(idx))):
            raise ValueError(
                f"{rule} {cell} ordering {int(ordering)} idx is not 0..{len(idx) - 1} without gaps"
            )
        if ledger is not None:
            expected = len(eligible_shas(ledger, str(cell)))
            if len(idx) != expected:
                raise ValueError(
                    f"{rule} {cell} ordering {int(ordering)} has {len(idx)} proposals; "
                    f"the ledger has {expected} registered candidates"
                )
        sequences.setdefault((str(cell), int(ordering)), {})[str(rule)] = tuple(ordered["sha"].astype(str))
        _check_chain(ordered, reference)
    for (cell, ordering), by_rule in sequences.items():
        unique = set(by_rule.values())
        if len(unique) > 1:
            raise ValueError(
                f"cell {cell} ordering {ordering} proposal sequence differs across rules "
                f"{sorted(by_rule)}"
            )


def _check_chain(group: pd.DataFrame, reference: str) -> None:
    running = reference
    accepted_rows = group.loc[group["accepted"].astype(bool)].sort_values(["idx", "sha"])
    for row in accepted_rows.itertuples(index=False):
        if str(row.incumbent) != running:
            raise ValueError(
                f"incumbent chain broken for rule={row.rule} cell={row.cell} "
                f"ordering={int(row.ordering)} idx={int(row.idx)}: "
                f"expected {running}, found {row.incumbent}"
            )
        running = str(row.sha)


def accepted(decisions: pd.DataFrame) -> pd.DataFrame:
    """Accepted, non-canary rows."""
    if "accepted" not in decisions.columns:
        raise ValueError("decisions have no accepted column")
    mask = decisions["accepted"].fillna(False).astype(bool)
    frame = decisions.loc[mask].copy()
    if frame.empty:
        return frame
    if "canary" in frame.columns:
        frame = frame.loc[~frame["canary"].fillna(False).astype(bool)].copy()
    return frame


def ordering_counts(decisions: pd.DataFrame) -> pd.DataFrame:
    counts = (
        decisions.groupby(["rule", "cell"], sort=True)["ordering"]
        .nunique()
        .rename("n_orderings")
        .reset_index()
    )
    return counts


def write_decision_tree(decisions: pd.DataFrame, root: str | Path) -> None:
    """Write ``root/<rule>/ord_NNN.jsonl`` for tests and the reference ratchet."""
    destination = Path(root)
    grouped = decisions.groupby(["rule", "ordering"], sort=True)
    for (rule, ordering), group in grouped:
        folder = destination / str(rule)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"ord_{int(ordering):03d}.jsonl"
        lines = []
        for row in group.sort_values(["cell", "idx"]).to_dict(orient="records"):
            payload = {}
            for key, value in row.items():
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    payload[key] = None
                elif hasattr(value, "item") and not isinstance(value, (str, bytes)):
                    payload[key] = value.item()
                else:
                    payload[key] = value
            lines.append(json.dumps(payload, sort_keys=True, allow_nan=False))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
