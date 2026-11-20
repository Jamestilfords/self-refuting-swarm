"""Experiment 1 tables from a committed unblind record.

``runs/unblind.json`` is required. Confirmatory tables use orderings 0–99.
When later orderings were published, ``tables/exploratory/`` holds the same
tables for the full log. Scorecard verdicts are written when the open score
files for the eight pairs are on disk.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pandas as pd

from skeptic.analysis.config import DEFAULT, AnalysisConfig
from skeptic.analysis.decisions import load_decisions, load_ledger
from skeptic.analysis.report_tables import write_all
from skeptic.analysis.scorecard import PAIRS, run_scorecard
from skeptic.analysis.sources import OpenSource, SealedNpzSource, file_sha256
from skeptic.analysis.truth import TruthCache, compute_truth_table, plan_pairs
from skeptic.store import UnsealPermit


def _summary(hypotheses_csv: Path, scorecard_note: str, registered_orderings: int) -> str:
    lines = [
        "# Experiment 1",
        "",
        f"Confirmatory orderings are 0–{registered_orderings - 1}. "
        "tables/exploratory holds every published ordering when the log continues past that.",
        "",
        scorecard_note,
        "",
    ]
    text = hypotheses_csv.read_text(encoding="utf-8").splitlines()
    body = "\n".join(line for line in text if not line.startswith("#"))
    for row in csv.DictReader(body.splitlines()):
        lines.append(
            f"{row['hypothesis']}: estimate={row['estimate']} "
            f"ci=[{row['ci_low']}, {row['ci_high']}] p_holm={row['p_holm']} reject={row['reject']}"
        )
    lines.append("")
    return "\n".join(lines)


def _scorecard_note(repo: Path, out_dir: Path) -> str:
    scores_root = repo / "runs" / "scores"
    ledger_path = repo / "runs" / "ledger.jsonl"
    baselines_path = repo / "proposals" / "baselines.json"
    if not scores_root.is_dir() or not ledger_path.is_file() or not baselines_path.is_file():
        return "Scorecard: open score files for the eight pairs are not on this machine."
    try:
        from skeptic.critic_inputs import load_cell

        cells = {}
        for pair in PAIRS:
            if pair.cell not in cells:
                cells[pair.cell] = load_cell(pair.cell, scores_root, ledger_path, baselines_path)
        scores = {}
        baselines = {}
        for pair in PAIRS:
            loaded = cells[pair.cell]
            baselines[pair.cell] = loaded.baselines
            for name in (pair.candidate, pair.incumbent):
                if name in loaded.baselines.scores:
                    scores[(pair.cell, name)] = loaded.baselines.scores[name]
                elif name in loaded.candidates:
                    scores[(pair.cell, name)] = loaded.candidates[name]
                else:
                    return f"Scorecard: {pair.cell} has no scores named {name}."
        rows = run_scorecard(scores, baselines)
    except (OSError, ValueError, KeyError, FileNotFoundError) as exc:
        return f"Scorecard: not scored ({exc})."
    path = out_dir / "scorecard.csv"
    columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return f"Scorecard: {len(rows)} pairs written to {path.name}."


def build_report(
    repo: Path,
    *,
    decisions: pd.DataFrame,
    ledger: pd.DataFrame,
    test_source,
    val_source,
    config: AnalysisConfig = DEFAULT,
    out_dir: Path | None = None,
    decisions_sha256: str = "",
    cache_dir: Path | None = None,
    workers: int = 1,
) -> dict:
    record = repo / "runs" / "unblind.json"
    if not record.is_file():
        raise FileNotFoundError("runs/unblind.json is missing; run make unblind first")
    destination = Path(out_dir) if out_dir is not None else repo / "tables"
    if cache_dir is None:
        cache_dir = Path(repo) / "runs" / "truth"
    written = write_all(
        destination, decisions=decisions, ledger=ledger,
        truth=_truth(
            decisions, ledger, test_source, config,
            cache_dir=Path(cache_dir), workers=workers,
        ),
        test_source=test_source, val_source=val_source, config=config,
        decisions_sha256=decisions_sha256,
    )
    note = _scorecard_note(repo, destination)
    summary = _summary(destination / "hypotheses.csv", note, config.registered_orderings)
    (destination / "SUMMARY.md").write_text(summary, encoding="utf-8")
    written["SUMMARY.md"] = destination / "SUMMARY.md"
    return written


def _truth(decisions, ledger, test_source, config, cache_dir: Path, workers: int = 1):
    taken = decisions.loc[decisions["accepted"].fillna(False).astype(bool)] if "accepted" in decisions.columns else decisions
    pairs = plan_pairs(taken, ledger, config=config, strict=False)
    pairs = sorted(pairs, key=lambda pair: (pair[2], pair[1], pair[0]))
    cache = TruthCache(cache_dir)
    if workers == 1:
        return compute_truth_table(
            pairs, source=test_source, config=config, strict=False, cache=cache, map_fn=map,
        )
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        return compute_truth_table(
            pairs, source=test_source, config=config, strict=False,
            cache=cache,
            map_fn=lambda function, jobs: pool.imap(function, jobs, chunksize=16),
        )


def verify_archive(sealed_root: Path, record: dict) -> None:
    """Every unblinded path matches its sha256, and no extra score file is present."""
    root = Path(sealed_root)
    problems = []
    files = record.get("files") or {}
    for relative, digest in files.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            problems.append(f"missing {relative}")
            continue
        if file_sha256(path) != digest:
            problems.append(f"hash {relative}")
    scores = root / "scores"
    if scores.is_dir():
        for path in sorted(scores.rglob("*.npz")):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                problems.append(f"symlink {relative}")
                continue
            if relative not in files:
                problems.append(f"extra {relative}")
    if problems:
        raise ValueError("sealed archive does not match unblind.json:\n" + "\n".join(problems))


def _sources_from_unblind(repo: Path, sealed_root: Path):
    record = json.loads((repo / "runs" / "unblind.json").read_text(encoding="utf-8"))
    permit = UnsealPermit(record["permit"]["digest"], record["permit"]["commit"])
    manifest = json.loads((repo / "runs" / "decision_manifest.json").read_text(encoding="utf-8"))
    ledger = load_ledger(repo / "runs" / "ledger.jsonl")
    arms = {(str(row.cell), str(row.sha)): str(row.arm)
            for row in ledger.itertuples(index=False) if getattr(row, "arm", None)}
    test = SealedNpzSource(
        sealed_root, permit, manifest["references"], arms=arms, repo=repo,
        verified=record.get("files") or {},
    )
    val = OpenSource(repo / "runs" / "scores", {})
    return ledger, test, val


def main(argv=None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--sealed", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--truth-cache", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args(argv)
    if args.workers < 0:
        print("--workers must be an integer >= 0", file=sys.stderr)
        return 1
    workers = (os.cpu_count() or 1) if args.workers == 0 else args.workers
    cache_dir = args.truth_cache if args.truth_cache is not None else args.repo / "runs" / "truth"
    record = args.repo / "runs" / "unblind.json"
    if not record.is_file():
        print("runs/unblind.json is missing; run make unblind first", file=sys.stderr)
        return 1
    if args.sealed is None:
        print("report needs --sealed pointing at the released score archive", file=sys.stderr)
        return 1
    payload = json.loads(record.read_text(encoding="utf-8"))
    try:
        verify_archive(args.sealed, payload)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    ledger, test, val = _sources_from_unblind(args.repo, args.sealed)
    decisions = load_decisions(args.repo / "runs" / "decisions", ledger)
    build_report(
        args.repo, decisions=decisions, ledger=ledger, test_source=test, val_source=val,
        decisions_sha256=payload["permit"]["digest"], out_dir=args.out,
        cache_dir=cache_dir, workers=workers,
    )
    print(json.dumps({"status": "reported", "out": str(args.out or (args.repo / "tables"))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
