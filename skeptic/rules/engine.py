"""E5 matched-order replay and deterministic decision-log writer.

No Ryu-controlled default is chosen here. A caller supplies the ordering
count/seed, R1 parameters and noise source, and the R2 verdict/state factory.
"""

from __future__ import annotations

import json
import filecmp
import inspect
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Callable

from skeptic.rules.common import (
    CELLS, FROZEN_RULES, PROPOSAL_STREAMS, ScoreArtifact, accept_gates_ok,
    matched_orderings, reference_block,
)
from skeptic.score_artifacts import auroc
from skeptic.rules.critic import Critic
from skeptic.rules.ratchet import Ratchet
from skeptic.rules.thresholdout import NoiseSource, Thresholdout
from skeptic.store import GuardError, HASH_RELATIVE, MANIFEST_RELATIVE, decision_digest

ScoreLoader = Callable[[str, str], ScoreArtifact]
NoiseFactory = Callable[[str, int], NoiseSource]
CriticFactory = Callable[[str, int, str], Critic]


def replay_cell(rows: dict[tuple[str, str], dict], cell: str, *, initial_sha: str,
                ordering_count: int, ordering_seed: int, score_loader: ScoreLoader,
                r1_threshold: float | None = None, r1_split_seed: int | None = None,
                noise_factory: NoiseFactory | None = None,
                critic_factory: CriticFactory | None = None,
                validated_aurocs: Mapping[str, list[float]] | None = None,
                r1_split_scores: Mapping[str, tuple] | None = None,
                selected_orderings: Iterable[tuple[int, tuple[str, ...]]] | None = None
                ) -> dict[tuple[str, int], list[dict]]:
    """Replay the same scored proposal order for every frozen rule in one cell."""
    baseline = rows.get((initial_sha, cell))
    if baseline is None or baseline.get("stream") != "baseline" or baseline.get("status") != "ok":
        raise ValueError("Initial incumbent must be a scored baseline in this cell")
    if (r1_threshold is None) != (r1_split_seed is None) or \
            (r1_threshold is None) != (noise_factory is None):
        raise ValueError("R1 needs threshold, split seed and noise factory together")
    orderings = (enumerate(matched_orderings(rows, cell, ordering_count, ordering_seed))
                 if selected_orderings is None else selected_orderings)
    out: dict[tuple[str, int], list[dict]] = defaultdict(list)
    artifacts: dict[str, ScoreArtifact] = {}

    def scores(sha: str) -> ScoreArtifact:
        if sha not in artifacts:
            artifacts[sha] = score_loader(cell, sha)
        return artifacts[sha]

    def ratchet_row(row):
        if row.get("status") != "ok":
            return row
        if validated_aurocs is None:
            artifact = scores(row["sha"])
            values = [auroc(seed[:artifact.n_in], seed[artifact.n_in:]) for seed in artifact.scores]
        else:
            values = validated_aurocs[row["sha"]]
        return {**row, "val": {**row["val"], "auroc": values}}

    for ordering, shas in orderings:
        r0 = Ratchet.from_baseline(ratchet_row(baseline))
        r1 = (Thresholdout.from_baseline(initial_sha, scores(initial_sha),
                                        threshold=r1_threshold, split_seed=r1_split_seed,
                                        noise=noise_factory(cell, ordering), split_scores=r1_split_scores)
              if noise_factory is not None else None)
        critics = {}
        if critic_factory is not None:
            for rule in FROZEN_RULES:
                if not rule.startswith("R2"):
                    continue
                critic = _new_critic(critic_factory, cell, ordering, initial_sha, rule)
                if not isinstance(critic, Critic):
                    raise ValueError("R2 factory must return a fresh Critic adapter")
                critics[rule] = critic
        for position, sha in enumerate(shas):
            row = rows[(sha, cell)]
            if row["stream"] not in PROPOSAL_STREAMS:
                raise ValueError("Baselines and canaries cannot enter the reported stream")
            prefix = {"schema_version": 1, "cell": cell, "ordering": ordering, "idx": position}
            out[("R0", ordering)].append({**prefix, **r0.step(ratchet_row(row))})
            if r1 is not None:
                artifact = scores(sha) if row.get("status") == "ok" else None
                out[("R1", ordering)].append({**prefix, **r1.step(row, artifact, position)})
            for rule, critic in critics.items():
                out[(rule, ordering)].append({**prefix, **critic.step(row, position)})
    return dict(out)


def _new_critic(factory, cell, ordering, initial, rule):
    """Pass the rule name when the factory accepts it. Each call is a new state."""
    code = getattr(factory, "__code__", None)
    if code is not None and (code.co_argcount >= 4 or code.co_flags & inspect.CO_VARARGS):
        return factory(cell, ordering, initial, rule)
    return factory(cell, ordering, initial)


def combine_cells(*cell_replays: dict[tuple[str, int], list[dict]]) -> dict[tuple[str, int], list[dict]]:
    combined: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for replay in cell_replays:
        for key, rows in replay.items():
            combined[key].extend(rows)
    for key in combined:
        combined[key].sort(key=lambda row: (row["cell"], row["idx"]))
    return dict(combined)


def write_decision_logs(repo_root: str | Path,
                        logs: Mapping[tuple[str, int], list[dict]] |
                              Iterable[tuple[tuple[str, int], list[dict]]],
                        *, ordering_count: int, metadata: dict | None = None) -> list[str]:
    """Spool complete logs for every frozen rule before any create-only output writes."""
    repo = Path(repo_root)
    if (repo / HASH_RELATIVE).exists():
        raise GuardError("Decision hash already exists; logs are frozen")
    if type(ordering_count) is not int or ordering_count <= 0:
        raise ValueError("Ordering count must be positive")
    expected_keys = {(rule, ordering) for rule in FROZEN_RULES
                     for ordering in range(ordering_count)}
    if isinstance(logs, Mapping) and set(logs) != expected_keys:
        raise ValueError("Decision logs must include R0, R1 and R2 for every ordering")
    expected_cells = set(CELLS)
    paths = []
    reference: dict[tuple[int, str], tuple[str, ...]] = {}
    proposal_sets: dict[str, set[str]] = {}
    entries = ((key, logs[key]) for key in sorted(logs)) if isinstance(logs, Mapping) else iter(logs)
    with tempfile.TemporaryDirectory(prefix="skeptic_decision_check_") as temp:
        prepared: set[str] = set()
        seen: set[tuple[str, int]] = set()
        for (rule, ordering), rows in entries:
            if (rule, ordering) not in expected_keys or (rule, ordering) in seen:
                raise ValueError("Decision logs must include R0, R1 and R2 once for every ordering")
            seen.add((rule, ordering))
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
                    if str(rule).startswith("R2") and row["verdict"] == "ACCEPT" and \
                            not accept_gates_ok(rule, row.get("gates")):
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
            path = Path(temp) / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as handle:
                for row in sorted(rows, key=lambda item: (item["cell"], item["idx"])):
                    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            prepared.add(rel)
        if seen != expected_keys:
            raise ValueError("Decision logs must include R0, R1 and R2 for every ordering")
        manifest = {"schema_version": 1, "rules": list(FROZEN_RULES),
                    "orderings": ordering_count, "cells": sorted(expected_cells),
                    "files": sorted(paths),
                    "proposals": {cell: sorted(shas) for cell, shas in sorted(proposal_sets.items())},
                    "references": reference_block(sorted(expected_cells))}
        if metadata is not None:
            manifest["replay"] = metadata
        manifest_path = Path(temp) / MANIFEST_RELATIVE
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                 encoding="utf-8")
        prepared.add(MANIFEST_RELATIVE.as_posix())
        # Apply the same complete schema checks as the freeze command before writing.
        decision_digest(temp)
        for rule in FROZEN_RULES:
            directory = repo / "runs/decisions" / rule
            if directory.exists() and any(p.is_file() and p.relative_to(repo).as_posix() not in prepared
                                          for p in directory.rglob("*")):
                raise GuardError("Unexpected pre-existing decision file")
        for rel in prepared:
            target = repo / rel
            if any(p.is_symlink() for p in (target, *target.parents) if p != repo.parent):
                raise GuardError("Decision paths cannot contain symlinks")
        # Check every existing file before writing any new one. A retry can
        # finish an interrupted write, but it cannot replace prior decisions.
        for rel in prepared:
            target = repo / rel
            if target.exists() and not filecmp.cmp(Path(temp) / rel, target, shallow=False):
                raise GuardError(f"Conflicting decision file: {rel}")
        for rel in sorted(prepared):
            target = repo / rel
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with (Path(temp) / rel).open("rb") as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
    return sorted(paths)


