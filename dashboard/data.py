"""Project only allowlisted validation fields from the committed demo inputs."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

CELLS = ("banking77_oos/bert", "banking77_oos/roberta", "clinc150/bert", "clinc150/roberta")
ROOT = Path(__file__).resolve().parents[1]
RUN_ID = re.compile(r"(?:banking77|clinc150)_bert/[ABC][0-2]\Z")


def read_local(root: Path, relative: str) -> Path:
    path = root / relative
    if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root.parent):
        raise ValueError("Dashboard inputs cannot be symlinks")
    return path


def json_rows(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # An active writer may not have finished the last JSONL record.
            if i != len(lines) - 1 or content.endswith("\n"):
                raise
    return rows


def label(cell: str) -> str:
    dataset, backbone = cell.split("/")
    return f"{'Banking77-OOS' if dataset == 'banking77_oos' else 'CLINC150'} / {'BERT' if backbone == 'bert' else 'RoBERTa'}"


def parse_report(text: str) -> dict:
    """Read James's published summary; do not infer missing replay results."""
    tables = {}
    headers = None
    for line in text.splitlines():
        if not line.startswith("|"):
            headers = None
            continue
        values = [v.strip() for v in line.strip("|").split("|")]
        if all(re.fullmatch(r"[-: ]+", v) for v in values):
            continue
        if headers is None:
            headers = values
        elif values[0] in CELLS:
            tables.setdefault(headers[1], {})[values[0]] = dict(zip(headers, values))
    match = re.search(r"\((\d+) orderings per cell\)", text)
    if not match:
        raise ValueError("Validation summary has no ordering count")
    cells = []
    for cell in CELLS:
        accepted = tables["Candidates"][cell]
        verdicts = tables["SCORE-MATCHED"][cell]
        best = tables["CE energy"][cell]
        ratchet = tables["Accepted steps"][cell]
        accepts = {}
        for rule, name in (("R0", "R0 ratchet"), ("R1", "R1 Thresholdout"), ("R2", "R2 critic")):
            values = re.fullmatch(r"([\d.]+) \[(\d+)-(\d+)\]", accepted[name])
            if not values:
                raise ValueError("Invalid acceptance summary")
            accepts[rule] = dict(zip(("mean", "min", "max"), map(float, values.groups())))
        baseline, baseline_auc = best["Strongest baseline"].rsplit(" ", 1)
        candidate, candidate_auc = best["Best candidate"].rsplit(" ", 1)
        cells.append({"id": cell, "label": label(cell), "candidates": int(accepted["Candidates"]),
                      "accepts": accepts,
                      "verdicts": {k: float(verdicts[k].rstrip("%")) for k in
                                   ("SCORE-MATCHED", "NO EFFECT", "UNDERPOWERED", "INERT", "other")},
                      "critic_accepted": int(verdicts["ACCEPT"]),
                      "ratchet": {"accepted_steps": int(ratchet["Accepted steps"]),
                                  "median_gain": float(ratchet["Median gain"]),
                                  "small_gains": float(ratchet["Gain < 0.001"].rstrip("%"))},
                      "best": {"sha": candidate.strip("`"), "auroc": float(candidate_auc),
                               "baseline": baseline, "baseline_auroc": float(baseline_auc),
                               "gap": float(best["Gap"]), "min_p": float(best["Smallest G3 p in the replay"])}})
    return {"orderings": int(match[1]), "cells": cells}


def snapshot(root: Path = ROOT) -> dict:
    report_path = read_local(root, "tables/validation_summary.md")
    report = parse_report(report_path.read_text())
    ledger = json_rows(read_local(root, "runs/ledger.jsonl"))
    baselines = json.loads(read_local(root, "proposals/baselines.json").read_text())
    baseline_shas = {r["sha"] for r in baselines}
    counts = {cell: sum(r["cell"] == cell and r["status"] == "ok" and r["stream"] in ("llm", "templated")
                        and r["sha"] not in baseline_shas for r in ledger) for cell in CELLS}
    canaries = json.loads(read_local(root, "proposals/canaries.json").read_text())
    runs = []
    summary_path = read_local(root, "runs/exp2/summary.json")
    for run in json.loads(summary_path.read_text()):
        run_id = "/".join(run["run"].replace("\\", "/").split("/")[-2:])
        if not RUN_ID.fullmatch(run_id):
            continue
        fields = ("arm", "cell", "slots", "judged", "accepted", "not_scored", "wasted_first_half",
                  "wasted_second_half", "wasted_change", "final_val_auroc")
        runs.append({"id": run_id, **{key: run.get(key) for key in fields}})
    source_paths = [report_path, summary_path, read_local(root, "runs/ledger.jsonl"),
                    read_local(root, "proposals/baselines.json"), read_local(root, "proposals/canaries.json")]
    signature = hashlib.sha256(b"".join(p.read_bytes() for p in source_paths)).hexdigest()
    return {"report": report, "runs": runs, "candidate_counts": counts,
            "ledger_proposals": len({r["sha"] for r in ledger}),
            "canaries": [{"name": c["name"], "sha": c["sha"], "expected": c["expected"], "why": c["why"],
                          "statuses": dict(Counter(r["status"] for r in ledger if r["sha"] == c["sha"]))} for c in canaries],
            "sources": [p.relative_to(root).as_posix() for p in source_paths],
            "updated_at": datetime.fromtimestamp(max(p.stat().st_mtime for p in source_paths), timezone.utc).isoformat(),
            "revision": signature, "validation_only": True}


def recorded_run(run_id: str, root: Path = ROOT) -> dict:
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("Unknown run")
    path = read_local(root, f"runs/exp2/{run_id}/decisions.jsonl")
    rows = json_rows(path)
    public = []
    for row in rows:
        keys = ("idx", "sha", "family", "verdict", "accepted", "status", "summary", "val_auroc", "delta_inc",
                "incumbent", "score_arm", "cell", "rule")
        item = {key: row.get(key) for key in keys}
        source = row.get("source") or ""
        parsed = urlsplit(source)
        item["source"] = source if parsed.scheme in ("https", "http") and parsed.netloc else ""
        critic = row.get("critic") or {}
        item["critic"] = {key: critic.get(key) for key in ("alpha_t", "ci", "comparator", "delta_val", "mdd", "p")}
        item["critic"]["gates"] = {key: value for key, value in (critic.get("gates") or {}).items()
                                   if key in ("G1", "G2", "G3", "G4", "G5", "G6", "lock")}
        item["memory_count"] = len(row.get("memory_used") or [])
        public.append(item)
    return {"id": run_id, "recorded": True, "rows": public}
