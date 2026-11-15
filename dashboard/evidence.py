"""Read-only joins for dashboard traces and local artifact health.

No scoring, provider calls, unblinding, or experiment-output writes occur here.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from dashboard.data import (CELLS, ROOT, RUN_ID, json_rows, parse_report,
                            public_text, read_local, recorded_run, web_url)

SHA = re.compile(r"[0-9a-f]{12}\Z")
MEMORY_ID = re.compile(r"m_[0-9]+\Z")
EXPECTED_RUNS = tuple(f"{group}/{arm}{seed}" for group in ("banking77_bert", "clinc150_bert")
                      for arm in "ABC" for seed in range(3))
INPUT_ERRORS = (ValueError, KeyError, TypeError, OSError)
NOTE_PATH = "tables/evidence_bundle.json"
PASTDEV_PATH = "tables/pastdev_check.json"


def _json(root, relative):
    return json.loads(read_local(root, relative).read_text(encoding="utf-8"))


def _metadata(root, relative):
    try:
        path = read_local(root, relative)
        stat = path.stat()
        if not path.is_file():
            raise ValueError("Not a regular file")
        return {"path": relative, "status": "Present", "bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}
    except FileNotFoundError:
        return {"path": relative, "status": "Missing", "bytes": None, "modified_at": None}
    except INPUT_ERRORS:
        return {"path": relative, "status": "Unreadable", "bytes": None, "modified_at": None}


def _ids(values, pattern):
    return [v for v in values if isinstance(v, str) and pattern.fullmatch(v)] if isinstance(values, list) else []


def discover_runs(root=ROOT):
    """Discover only named validation-run directories; never follow symlinks."""
    found = []
    for group in ("banking77_bert", "clinc150_bert"):
        parent = read_local(root, f"runs/exp2/{group}")
        if not parent.exists():
            continue
        for path in parent.iterdir():
            run_id = f"{group}/{path.name}"
            if RUN_ID.fullmatch(run_id) and not path.is_symlink() and path.is_dir():
                found.append(run_id)
    return sorted(found)


def demo_notes(root=ROOT):
    """Optional editorial handoff; notes never replace measured results."""
    try:
        raw = _json(root, NOTE_PATH)
        if not isinstance(raw, dict):
            raise ValueError("Unsupported evidence bundle")
        research = raw["explanations"]
        notes = {k: public_text(research[source], 3000) for k, source in
                 (("question", "research_question"), ("observed_result", "observed_result"), ("limitation", "limitation"))}
        if not all(notes.values()):
            raise ValueError("Missing research explanation")
        example = raw.get("example") or {}
        if not isinstance(example, dict):
            raise ValueError("Invalid example")
        proposal = example.get("proposal") or {}
        if not isinstance(proposal, dict):
            raise ValueError("Invalid proposal")
        candidates = [example.get("run_id"), example.get("run"), example.get("run_dir"), proposal.get("file"), proposal.get("path"), proposal.get("file_path")]
        run_id = next((v for v in candidates if isinstance(v, str) and RUN_ID.fullmatch(v)), None)
        for value in candidates:
            if not run_id and isinstance(value, str):
                match = re.search(r"(?:^|/)runs/exp2/((?:banking77|clinc150)_bert/[ABC][0-9]+)(?:/|$)", value)
                if match and RUN_ID.fullmatch(match[1]):
                    run_id = match[1]
        featured = {"run_id": run_id, "proposal_sha": proposal.get("sha")} if run_id and SHA.fullmatch(str(proposal.get("sha", ""))) else {}
        sponsors = {}
        raw_sponsors = raw.get("sponsor_evidence") or {}
        if not isinstance(raw_sponsors, dict):
            raise ValueError("Invalid sponsor evidence")
        for name in ("Exa", "Modal", "past.dev"):
            value = raw_sponsors.get(name) or raw_sponsors.get(name.lower()) or (raw_sponsors.get("pastdev") if name == "past.dev" else None)
            if isinstance(value, dict):
                sponsors[name] = {"status": public_text(value.get("status"), 150), "evidence": public_text(value.get("evidence"), 3000)}
        return {"status": "Provided", "research": notes,
                "featured_trace": {k: featured[k] for k in ("run_id", "proposal_sha") if k in featured},
                "prepared_by": public_text(raw.get("prepared_by"), 100), "path": NOTE_PATH,
                "sponsors": sponsors}
    except FileNotFoundError:
        return {"status": "Not supplied", "research": None, "featured_trace": {}, "path": NOTE_PATH}
    except INPUT_ERRORS:
        return {"status": "Invalid", "research": None, "featured_trace": {}, "path": NOTE_PATH}


def memory_usage(root=ROOT):
    """Count direct declared references; proposal counts and citation counts are different."""
    counts = []
    for run_id in EXPECTED_RUNS:
        try:
            prefix = f"runs/exp2/{run_id}"
            entries = json_rows(read_local(root, f"{prefix}/memory.jsonl"))
            kinds = {e["id"]: e["kind"] for e in entries}
            rows = recorded_run(run_id, root)["rows"]
            judged = [r for r in rows if r["status"] == "judged"]
            refs = [[kinds.get(mid) for mid in r["memory_used"]] for r in judged]
            counts.append({"run": run_id, "arm": run_id.split("/")[-1][0], "available": True,
                           "source_cards": sum(k == "source" for k in kinds.values()),
                           "lessons": sum(k == "lesson" for k in kinds.values()), "judged": len(judged),
                           "source_proposals": sum("source" in used for used in refs),
                           "source_citations": sum(used.count("source") for used in refs),
                           "lesson_proposals": sum("lesson" in used for used in refs),
                           "unknown_references": sum(used.count(None) for used in refs)})
        except INPUT_ERRORS:
            counts.append({"run": run_id, "arm": run_id.split("/")[-1][0], "available": False})
    return counts


def trace_run(run_id, root=ROOT):
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("Unknown run")
    prefix = f"runs/exp2/{run_id}"
    decisions = recorded_run(run_id, root)["rows"]
    warnings = []
    try:
        memories = json_rows(read_local(root, f"{prefix}/memory.jsonl"))
    except INPUT_ERRORS:
        memories = []
        warnings.append("Memory file missing or invalid; source and lesson links are unavailable.")
    try:
        manifest = _json(root, f"{prefix}/proposals/llm.json")
        if not isinstance(manifest, dict) or any(not isinstance(manifest.get(key, []), list) for key in ("proposals", "rejected")):
            raise ValueError("Invalid manifest")
    except INPUT_ERRORS:
        manifest = {}
        warnings.append("Proposal manifest missing or invalid; model and attempt counts may be unavailable.")
    sources, proposals, lessons, memory_by_id = {}, {}, [], {}
    for entry in memories:
        mid, kind, content = entry.get("id"), entry.get("kind"), entry.get("content")
        if not isinstance(mid, str) or not MEMORY_ID.fullmatch(mid) or not isinstance(content, dict):
            continue
        memory_by_id[mid] = {"id": mid, "kind": public_text(kind, 30), "author": public_text(entry.get("author"), 40),
                             "sha": content.get("sha") if SHA.fullmatch(str(content.get("sha", ""))) else None,
                             "memory_used": _ids(content.get("memory_used"), MEMORY_ID)}
        if kind == "source":
            sources[mid] = {k: public_text(content.get(k)) for k in
                            ("title", "query", "method", "date", "target_failure", "excerpt_sha256")}
            sources[mid].update(id=mid, url=web_url(content.get("url")), author=public_text(entry.get("author"), 40))
        elif kind == "proposal" and SHA.fullmatch(str(content.get("sha", ""))):
            proposals[content["sha"]] = {"memory_id": mid, "rationale": public_text(content.get("rationale")),
                                        "memory_used": _ids(content.get("memory_used"), MEMORY_ID)}
        elif kind == "lesson":
            lessons.append({"id": mid, "text": public_text(content.get("text")),
                            "cites": _ids(content.get("cites"), SHA), "after": content.get("after"),
                            "author": public_text(entry.get("author"), 40)})
    # Retain decisions as the authority for timing, outcome, and actually declared memory use.
    manifest_by_sha = {r.get("sha"): r for r in manifest.get("proposals", []) if isinstance(r, dict)}
    rejected_by_idx = {r.get("idx"): r for r in manifest.get("rejected", []) if isinstance(r, dict)}
    cards = []
    for position, decision in enumerate(decisions):
        card = dict(decision, slot=position + 1)
        sha = decision["sha"]
        proposal = proposals.get(sha, {})
        metadata = manifest_by_sha.get(sha, {})
        card.update(rationale=proposal.get("rationale", ""),
                    proposal_memory_id=proposal.get("memory_id"),
                    model=public_text(metadata.get("model"), 100), attempts=metadata.get("attempts"))
        card["failure_detail"] = public_text(rejected_by_idx.get(decision["idx"], {}).get("detail"))
        ancestry, visited = [], set()
        queue = deque((mid, [mid]) for mid in decision["memory_used"])
        while queue and len(visited) < 1000:
            mid, path = queue.popleft()
            if mid in visited:
                continue
            visited.add(mid)
            entry = memory_by_id.get(mid)
            if not entry:
                continue
            ancestry.append({k: entry[k] for k in ("id", "kind", "author", "sha")} | {"via_ids": path})
            if len(path) < 40:
                queue.extend((child, path + [child]) for child in entry["memory_used"] if child not in visited)
        card["memory_ancestry"] = ancestry
        paths = {entry["id"]: entry["via_ids"] for entry in ancestry}
        linked = []
        for mid, source in sources.items():
            # Both explicit MEMORY_USED and an exact SOURCE URL support a link; distinguish them.
            if mid in paths or (source["url"] and source["url"] == decision["source"]):
                link = "MEMORY_USED" if mid in decision["memory_used"] else "Indirect memory ancestry" if mid in paths else "SOURCE URL"
                linked.append(dict(source, link=link, via_ids=paths.get(mid, [])))
        card["sources"] = linked
        card["used_lessons"] = [dict(l, via_ids=paths[l["id"]], direct=l["id"] in decision["memory_used"])
                                for l in lessons if l["id"] in paths]
        card["resulting_lessons"] = []
        for lesson in lessons:
            if sha not in lesson["cites"]:
                continue
            later = [{"sha": d["sha"], "slot": i + 1, "verdict": d["verdict"], "family": d["family"],
                      "delta_inc": d["delta_inc"]} for i, d in enumerate(decisions)
                     if i > position and lesson["id"] in d["memory_used"]]
            card["resulting_lessons"].append(dict(lesson, followups=later))
        cards.append(card)
    cards_by_sha = {c["sha"]: c for c in cards if c["sha"]}
    for card in cards:
        for lesson in card["used_lessons"]:
            earlier = [cards_by_sha[sha] for sha in lesson["cites"] if sha in cards_by_sha and cards_by_sha[sha]["slot"] < card["slot"]]
            lesson["built_from_verdicts_on"] = [{k: c[k] for k in ("sha", "slot", "verdict", "reason", "source", "rationale", "summary")} for c in earlier]
        card["complete_trace"] = bool(card["sources"] and (any(l["followups"] for l in card["resulting_lessons"])
                                      or any(l["built_from_verdicts_on"] for l in card["used_lessons"])))
    complete = [c for c in cards if c["complete_trace"]]
    featured = complete[0]["slot"] if complete else (cards[0]["slot"] if cards else None)
    return {"run_id": run_id, "cards": cards, "featured_slot": featured, "warnings": warnings,
            "source_count": len(sources), "lesson_count": len(lessons), "complete_traces": len(complete),
            "memory_transport": "Local artifacts; successful past.dev reads are not established by these logs.",
            "files": [_metadata(root, f"{prefix}/{name}") for name in
                      ("decisions.jsonl", "memory.jsonl", "proposals/llm.json", "budget.json", "scout.json")]}


def proposal_source(run_id, sha, root=ROOT):
    """Display source as text; never import or execute generated proposal code."""
    if not RUN_ID.fullmatch(run_id) or not SHA.fullmatch(sha):
        raise ValueError("Invalid proposal identity")
    relative = f"runs/exp2/{run_id}/proposals/{sha}.py"
    try:
        path = read_local(root, relative)
        if path.stat().st_size > 256_000:
            raise ValueError("Proposal too large")
        raw = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(raw.encode()).hexdigest()
        rationale = ""
        try:
            tree = ast.parse(raw)
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RATIONALE" for t in node.targets):
                    value = ast.literal_eval(node.value)
                    rationale = public_text(value)
        except (SyntaxError, ValueError, TypeError):
            pass
        return {"status": "Matched" if digest[:12] == sha else "Hash mismatch", "code": public_text(raw, 256_000),
                "rationale": rationale, "sha256": digest, "path": relative}
    except FileNotFoundError:
        return {"status": "Missing", "code": "", "rationale": "", "path": relative}
    except INPUT_ERRORS:
        return {"status": "Unreadable", "code": "", "rationale": "", "path": relative}


def run_health(root=ROOT):
    """Compare local artifacts to registered counts; existence is not integrity verification."""
    checks, runs, failures = [], [], []

    def check(name, actual, expected, detail=""):
        checks.append({"check": name, "observed": actual, "expected": expected,
                       "status": "Unavailable" if actual is None else "Match" if actual == expected else "Mismatch",
                       "detail": detail})

    ledger = None
    try:
        ledger = json_rows(read_local(root, "runs/ledger.jsonl"))
        keys = [(r["sha"], r["cell"]) for r in ledger]
        check("Unique proposal IDs", len({s for s, _ in keys}), 500)
        check("Scoring ledger rows", len(ledger), 2000)
        check("Duplicate proposal/cell rows", len(keys) - len(set(keys)), 0)
        check("Unknown scoring cells", sum(cell not in CELLS for _, cell in keys), 0)
        groups = {}
        for sha, cell in keys:
            groups.setdefault(sha, set()).add(cell)
        check("Proposals covering all four cells", sum(cells == set(CELLS) for cells in groups.values()), 500)
        check("Successful scoring rows", sum(r["status"] == "ok" for r in ledger), 1996)
        check("I/O-blocked scoring rows", sum(r["status"] == "io" for r in ledger), 4,
              "Check Guard & provenance for the named leak canary; an I/O status is not a critic verdict.")
        try:
            report = parse_report(read_local(root, "tables/validation_summary.md").read_text())
            bases = {r["sha"] for r in _json(root, "proposals/baselines.json")}
            for cell in report["cells"]:
                observed = len({r["sha"] for r in ledger if r["cell"] == cell["id"] and r["status"] == "ok"
                                and r["stream"] in ("llm", "templated") and r["sha"] not in bases})
                check(f"Candidates: {cell['id']}", observed, cell["candidates"], "Ledger versus published summary")
        except INPUT_ERRORS:
            check("Summary versus ledger", None, "Readable summary and baseline registry")
    except INPUT_ERRORS:
        check("Scoring ledger", None, "2,000 readable rows")
    try:
        found = discover_runs(root)
    except INPUT_ERRORS:
        found = []
    try:
        summary = {"/".join(r["run"].replace("\\", "/").split("/")[-2:]): r for r in _json(root, "runs/exp2/summary.json")}
    except INPUT_ERRORS:
        summary = {}
    total_slots, all_decisions_available = 0, True
    for run_id in sorted(set(EXPECTED_RUNS) | set(found)):
        prefix = f"runs/exp2/{run_id}"
        item = {"run": run_id, "registered": run_id in EXPECTED_RUNS, "slots": None,
                "judged": None, "unscored": None, "accepted": None, "problems": [], "usage": {}, "limits": {}}
        try:
            rows = recorded_run(run_id, root)["rows"]
            item.update(slots=len(rows), judged=sum(r["status"] == "judged" for r in rows),
                        unscored=sum(r["status"] != "judged" for r in rows), accepted=sum(bool(r["accepted"]) for r in rows))
            if item["registered"]:
                total_slots += len(rows)
                if len(rows) != 80:
                    item["problems"].append("Expected 80 registered slots")
            if [r["idx"] for r in rows] != list(range(len(rows))):
                item["problems"].append("Slot indices are missing, duplicated or out of order")
            if any(not SHA.fullmatch(str(r["sha"])) for r in rows if r["status"] == "judged"):
                item["problems"].append("Invalid judged proposal identifiers")
            saved = summary.get(run_id)
            if not saved:
                item["problems"].append("Not listed in summary")
            elif any(saved.get(key) != item[key] for key in ("slots", "judged", "accepted")) or saved.get("not_scored") != item["unscored"]:
                item["problems"].append("Decision counts differ from summary")
            for row in rows:
                if row["status"] != "judged":
                    failures.append({"run": run_id, "slot": (row["idx"] + 1) if isinstance(row["idx"], int) else None,
                                     "reason": row["reason"] or "Not recorded"})
            missing_code = sum(not read_local(root, f"{prefix}/proposals/{r['sha']}.py").is_file()
                               for r in rows if r["status"] == "judged" and SHA.fullmatch(str(r["sha"])))
            if missing_code:
                item["problems"].append(f"{missing_code} proposal source files missing")
        except INPUT_ERRORS:
            item["problems"].append("Decision file missing or invalid")
            if item["registered"]:
                all_decisions_available = False
        for name in ("memory.jsonl", "proposals/llm.json", "budget.json", "state.json"):
            meta = _metadata(root, f"{prefix}/{name}")
            if meta["status"] != "Present":
                item["problems"].append(f"{name}: {meta['status'].lower()}")
        try:
            budget = _json(root, f"{prefix}/budget.json")
            if not isinstance(budget, dict) or any(not isinstance(budget.get(key), dict) for key in ("used", "limits")):
                raise ValueError("Invalid budget")
            for section, keys in (("used", ("calls", "searches", "charged_tokens", "input_tokens", "output_tokens")),
                                  ("limits", ("max_calls", "max_searches", "max_tokens"))):
                target = "usage" if section == "used" else "limits"
                item[target] = {key: value for key in keys if type(value := budget[section].get(key)) is int and value >= 0}
                if len(item[target]) != len(keys):
                    item["problems"].append(f"Incomplete {section} in budget.json")
        except INPUT_ERRORS:
            item["problems"].append("Budget unreadable")
        item["modified_at"] = _metadata(root, f"{prefix}/decisions.jsonl")["modified_at"]
        item["status"] = "Needs attention" if item["problems"] else "Counts match"
        runs.append(item)
    check("Registered run directories", len(set(found) & set(EXPECTED_RUNS)), 18)
    check("Recorded registered slots", total_slots if all_decisions_available else None, 1440)
    check("Runs matching summary and file checks", sum(r["status"] == "Counts match" for r in runs if r["registered"]), 18)
    artifacts = [_metadata(root, p) for p in ("tables/validation_summary.md", "runs/ledger.jsonl", "runs/exp2/summary.json",
                 "runs/score_manifest.sha256", "runs/decision_manifest.json", "runs/DECISIONS.sha256", "runs/guard.log", NOTE_PATH, PASTDEV_PATH)]
    # Bounded metadata checks only. No scoring arrays, freeze procedure, or sealed-test files are read.
    for rule in ("R0", "R1", "R2"):
        count = sum(_metadata(root, f"runs/decisions/{rule}/ord_{i:03d}.jsonl")["status"] == "Present" for i in range(200))
        check(f"Detailed {rule} replay logs", count, 200, "File presence only; hashes and row coverage not verified")
    return {"checks": checks, "runs": runs, "failures": failures, "artifacts": artifacts,
            "local_only": True, "training_verified": False, "freeze_verified": False,
            "checked_at": datetime.now(timezone.utc).isoformat()}
