"""E5 matched-order replay and deterministic decision-log writer.

No Ryu-controlled default is chosen here. A caller supplies the ordering
count/seed, R1 parameters and noise source, and the R2 verdict/state factory.
"""

from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Callable

from skeptic.rules.common import CELLS, PROPOSAL_STREAMS, ScoreArtifact, matched_orderings
from skeptic.score_artifacts import auroc
from skeptic.rules.critic import Critic
from skeptic.rules.ratchet import Ratchet
from skeptic.rules.thresholdout import NoisePair, Thresholdout
from skeptic.store import GuardError, HASH_RELATIVE, MANIFEST_RELATIVE, decision_digest

ScoreLoader = Callable[[str, str], ScoreArtifact]
NoiseFactory = Callable[[str, int], NoisePair]
CriticFactory = Callable[[str, int, str], Critic]


def replay_cell(rows: dict[tuple[str, str], dict], cell: str, *, initial_sha: str,
                ordering_count: int, ordering_seed: int, score_loader: ScoreLoader,
                r1_threshold: float | None = None, r1_split_seed: int | None = None,
                noise_factory: NoiseFactory | None = None,
                critic_factory: CriticFactory | None = None) -> dict[tuple[str, int], list[dict]]:
    """Replay matched orderings for one cell; R1/R2 can be added as their contracts land."""
    baseline = rows.get((initial_sha, cell))
    if baseline is None or baseline.get("stream") != "baseline" or baseline.get("status") != "ok":
        raise ValueError("Initial incumbent must be a scored baseline in this cell")
    if (r1_threshold is None) != (r1_split_seed is None) or \
            (r1_threshold is None) != (noise_factory is None):
        raise ValueError("R1 needs threshold, split seed and noise factory together")
    orderings = matched_orderings(rows, cell, ordering_count, ordering_seed)
    out: dict[tuple[str, int], list[dict]] = defaultdict(list)
    artifacts: dict[str, ScoreArtifact] = {}

    def scores(sha: str) -> ScoreArtifact:
        if sha not in artifacts:
            artifacts[sha] = score_loader(cell, sha)
        return artifacts[sha]

    def ratchet_row(row):
        if row.get("status") != "ok":
            return row
        artifact = scores(row["sha"])
        return {**row, "val": {**row["val"], "auroc": [
            auroc(seed[:artifact.n_in], seed[artifact.n_in:]) for seed in artifact.scores]}}

    for ordering, shas in enumerate(orderings):
        r0 = Ratchet.from_baseline(ratchet_row(baseline))
        r1 = (Thresholdout.from_baseline(initial_sha, scores(initial_sha),
                                        threshold=r1_threshold, split_seed=r1_split_seed,
                                        noise=noise_factory(cell, ordering))
              if noise_factory is not None else None)
        r2 = critic_factory(cell, ordering, initial_sha) if critic_factory is not None else None
        if r2 is not None and not isinstance(r2, Critic):
            raise ValueError("R2 factory must return a fresh Critic adapter")
        for position, sha in enumerate(shas):
            row = rows[(sha, cell)]
            if row["stream"] not in PROPOSAL_STREAMS:
                raise ValueError("Baselines and canaries cannot enter the reported stream")
            prefix = {"schema_version": 1, "cell": cell, "ordering": ordering, "idx": position}
            out[("R0", ordering)].append({**prefix, **r0.step(ratchet_row(row))})
            if r1 is not None:
                artifact = scores(sha) if row.get("status") == "ok" else None
                out[("R1", ordering)].append({**prefix, **r1.step(row, artifact, position)})
            if r2 is not None:
                out[("R2", ordering)].append({**prefix, **r2.step(row, position)})
    return dict(out)


def combine_cells(*cell_replays: dict[tuple[str, int], list[dict]]) -> dict[tuple[str, int], list[dict]]:
    combined: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for replay in cell_replays:
        for key, rows in replay.items():
            combined[key].extend(rows)
    for key in combined:
        combined[key].sort(key=lambda row: (row["cell"], row["idx"]))
    return dict(combined)


def write_decision_logs(repo_root: str | Path, logs: dict[tuple[str, int], list[dict]],
                        *, ordering_count: int, metadata: dict | None = None) -> list[str]:
    """Write the complete R0/R1/R2 set once; never overwrite a frozen decision."""
    repo = Path(repo_root)
    if (repo / HASH_RELATIVE).exists():
        raise GuardError("Decision hash already exists; logs are frozen")
    if type(ordering_count) is not int or ordering_count <= 0:
        raise ValueError("Ordering count must be positive")
    expected_keys = {(rule, ordering) for rule in ("R0", "R1", "R2")
                     for ordering in range(ordering_count)}
    if set(logs) != expected_keys:
        raise ValueError("Decision logs must include R0, R1 and R2 for every ordering")
    expected_cells = set(CELLS)
    paths = []
    prepared: dict[str, str] = {}
    reference: dict[tuple[int, str], tuple[str, ...]] = {}
    proposal_sets: dict[str, set[str]] = {}
    for rule, ordering in sorted(logs):
        rows = logs[(rule, ordering)]
        if not rows or {row.get("cell") for row in rows} != expected_cells \
                or any(row.get("rule") != rule or row.get("ordering") != ordering
                       or row.get("schema_version") != 1 for row in rows):
            raise ValueError("Decision log is incomplete or mislabeled")
        for cell in expected_cells:
            selected = [row for row in rows if row["cell"] == cell]
            positions = [row["idx"] for row in selected]
            if sorted(positions) != list(range(len(positions))):
                raise ValueError("Decision log has missing or duplicate positions")
            selected.sort(key=lambda row: row["idx"])
            sequence = tuple(row["sha"] for row in selected)
            incumbent = selected[0]["incumbent"]
            for row in selected:
                if row["incumbent"] != incumbent:
                    raise ValueError("Decision log has a broken incumbent chain")
                if rule in ("R0", "R1") and row["verdict"] not in ("ACCEPT", "REJECT"):
                    raise ValueError("R0/R1 decision has an invalid verdict")
                if rule == "R2" and row["verdict"] == "ACCEPT" and \
                        not _r2_gates_pass(row.get("gates")):
                    raise ValueError("R2 cannot accept unless G1/G2/G3/G5/G6 pass")
                if row["verdict"] == "ACCEPT":
                    incumbent = row["sha"]
            key = (ordering, cell)
            if key in reference and sequence != reference[key]:
                raise ValueError("Rules must replay the same proposal ordering")
            reference[key] = sequence
            if cell in proposal_sets and set(sequence) != proposal_sets[cell]:
                raise ValueError("Every ordering must use the same frozen proposal set")
            proposal_sets[cell] = set(sequence)
        rel = f"runs/decisions/{rule}/ord_{ordering:03d}.jsonl"
        paths.append(rel)
        data = "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                       for row in sorted(rows, key=lambda item: (item["cell"], item["idx"])))
        prepared[rel] = data
    manifest = {"schema_version": 1, "rules": ["R0", "R1", "R2"],
                "orderings": ordering_count, "cells": sorted(expected_cells),
                "files": sorted(paths),
                "proposals": {cell: sorted(shas) for cell, shas in sorted(proposal_sets.items())}}
    if metadata is not None:
        manifest["replay"] = metadata
    manifest_data = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    prepared[MANIFEST_RELATIVE.as_posix()] = manifest_data

    # Apply the same complete schema checks as the freeze command before writing.
    with tempfile.TemporaryDirectory(prefix="skeptic_decision_check_") as temp:
        for rel, data in prepared.items():
            path = Path(temp) / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data, encoding="utf-8")
        decision_digest(temp)
    for rule in ("R0", "R1", "R2"):
        directory = repo / "runs/decisions" / rule
        if directory.exists() and any(p.is_file() and p.relative_to(repo).as_posix() not in prepared
                                      for p in directory.rglob("*")):
            raise GuardError("Unexpected pre-existing decision file")
    for rel in prepared:
        target = repo / rel
        if any(p.is_symlink() for p in (target, *target.parents) if p != repo.parent):
            raise GuardError("Decision paths cannot contain symlinks")

    # Check every existing file before writing any new one. A retry can finish
    # an interrupted write, but it cannot replace a prior decision.
    for rel, data in prepared.items():
        target = repo / rel
        if target.exists():
            if target.is_symlink() or target.read_text(encoding="utf-8") != data:
                raise GuardError(f"Conflicting decision file: {rel}")
    for rel, data in prepared.items():
        target = repo / rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as handle:
                handle.write(data)
    return sorted(paths)


def _r2_gates_pass(gates: object) -> bool:
    if not isinstance(gates, dict):
        return False
    return all(gates.get(gate) == "pass" for gate in ("G1", "G2", "G3", "G5", "G6"))
