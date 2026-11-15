"""Validation-only evidence bundle: one proposal traced through the loop, plus counts and sponsor evidence.

    python -m skeptic.analysis.evidence_bundle export                 # writes tables/evidence_bundle.json
    python -m skeptic.analysis.evidence_bundle pastdev-check          # optional: two read-only recalls, saved as evidence

Everything is read from files already in the repository (runs/exp2, runs/ledger.jsonl, tables). No test
data, no scoring and no LLM call are involved.
"""
from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

EXPLANATIONS = {
    "research_question": "An automated research loop that keeps whatever scores higher on validation will always find "
                         "something to keep. Are those kept changes real improvements, or noise from trying many things?",
    "observed_result": "On the same 487 AI-written and templated proposals, replayed in 200 random orders per cell, the "
                       "keep-if-better ratchet accepted 5.4 to 6.6 proposals per ordering (ordering alone gives about 6.8), "
                       "Thresholdout accepted 29 to 66, and the statistical critic accepted none. In the live closed loop "
                       "(1,440 proposals) ratchet runs kept 25 proposals and critic runs kept 0.",
    "limitation": "All numbers are from validation data. The sealed test set was never opened, so false-discovery rate, "
                  "recall and hypotheses H1 to H4 are not computed, and we cannot say the ratchet's accepted proposals are "
                  "false. The critic may also be too strict: zero acceptances leave no eligible paired comparison. "
                  "Agents used each other's output, but the memory measure (H5) showed no benefit.",
}


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _header(path) -> dict:
    out = {}
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def _proposal(run: Path, decision: dict, memory: dict) -> dict:
    head = _header(run / "proposals" / f"{decision['sha']}.py")
    return {"sha": decision["sha"], "slot": decision["idx"], "family": decision["family"], "score_arm": decision["score_arm"],
            "rationale": head.get("RATIONALE", ""), "source_url": decision.get("source", ""),
            "memory_used": [{"id": m, "kind": memory[m]["kind"], "written_by": memory[m]["author"]}
                            for m in decision.get("memory_used", []) if m in memory],
            "file": f"{run.as_posix()}/proposals/{decision['sha']}.py", "memory_id": decision.get("memory_id")}


def _validation(decision: dict) -> dict:
    critic = decision.get("critic") or {}
    failed = next((g for g, v in (critic.get("gates") or {}).items() if v == "fail"), None)
    return {"validation_auroc_seed_mean": decision["val_auroc"], "gain_over_incumbent": decision["delta_inc"],
            "incumbent": decision["incumbent"], "rule": decision["rule"], "comparator": critic.get("comparator"),
            "gain_over_comparator": critic.get("delta_val"), "ci95": critic.get("ci"), "p": critic.get("p"),
            "lord_level": critic.get("alpha_t"), "gates": critic.get("gates"), "failed_gate": failed,
            "verdict": decision["verdict"], "accepted": decision["accepted"], "failure_reason": decision["summary"]}


def trace(run, sha: str) -> dict:
    """Source cards and lesson a proposal used, the earlier proposals behind that lesson, and what followed."""
    run = Path(run)
    memory = {e["id"]: e for e in _jsonl(run / "memory.jsonl")}
    decisions = [d for d in _jsonl(run / "decisions.jsonl") if d["status"] == "judged"]
    by_sha = {d["sha"]: d for d in decisions}
    d = by_sha[sha]
    used = [memory[m] for m in d.get("memory_used", []) if m in memory]
    sources = [{"id": e["id"], "written_by": e["author"], **{k: e["content"].get(k) for k in
                ("query", "title", "url", "date", "method", "target_failure")}} for e in used if e["kind"] == "source"]
    lessons = []
    for e in (x for x in used if x["kind"] == "lesson"):
        cited = [by_sha[s] for s in e["content"].get("cites", []) if s in by_sha]
        lessons.append({"id": e["id"], "written_by": e["author"], "after_judged_proposals": e["content"].get("after"),
                        "text": e["content"]["text"],
                        "built_from_verdicts_on": [{"sha": c["sha"], "slot": c["idx"], "family": c["family"], "verdict": c["verdict"],
                                                    "failure_reason": c["summary"], "source_url": c.get("source", ""),
                                                    "rationale": _header(run / "proposals" / f"{c['sha']}.py").get("RATIONALE", "")}
                                                   for c in cited]})
    mine = {d.get("memory_id")} | {e["id"] for e in memory.values()
                                   if e["kind"] == "verdict" and e["content"].get("sha") == sha}
    lesson_ids = {entry["id"] for entry in lessons}
    follow = [{**_proposal(run, x, memory), "verdict": x["verdict"], "failure_reason": x["summary"],
               "cites_this_proposal": bool(mine & set(x.get("memory_used", []))),
               "cites_same_lesson": bool(lesson_ids & set(x.get("memory_used", [])))}
              for x in decisions if x["idx"] > d["idx"] and (mine | lesson_ids) & set(x.get("memory_used", []))]
    return {"run": run.as_posix(), "cell": d["cell"], "arm": d["arm"],
            "steps": ["scout: Exa search -> source card", "critic: verdicts on earlier proposals", "analyst: lesson",
                      "proposer: this proposal, citing the source card and the lesson", "modal: validation scores, 5 seeds",
                      "critic: verdict"],
            "sources": sources, "lessons": lessons, "proposal": _proposal(run, d, memory), "validation": _validation(d),
            "later_proposals_citing_it_or_the_lesson": follow}


def memory_arm_counts(root="runs/exp2") -> list[dict]:
    rows = []
    for run in sorted(Path(root).glob("*/B*")):
        memory = {e["id"]: e for e in _jsonl(run / "memory.jsonl")}
        judged = [d for d in _jsonl(run / "decisions.jsonl") if d["status"] == "judged"]
        kinds = lambda d: {memory[m]["kind"] for m in d.get("memory_used", []) if m in memory}
        count = lambda kind: sum(1 for e in memory.values() if e["kind"] == kind)
        rows.append({"run": run.as_posix(), "source_cards_by_scout": count("source"), "lessons_by_analyst": count("lesson"),
                     "verdicts_by_critic": count("verdict"), "proposals_by_proposer": count("proposal"),
                     "judged_proposals": len(judged),
                     "judged_citing_a_source_card": sum("source" in kinds(d) for d in judged),
                     "judged_citing_a_lesson": sum("lesson" in kinds(d) for d in judged),
                     "judged_citing_a_verdict": sum("verdict" in kinds(d) for d in judged)})
    return rows


def sponsor_evidence(counts: list[dict]) -> dict:
    return {
        "exa": {"status": "verified from repository artifacts",
                "evidence": f"{sum(c['source_cards_by_scout'] for c in counts)} source cards written by the Scout in the six "
                            "memory-arm runs, each with the Exa query, title, URL, date and extracted method "
                            "(runs/exp2/*/B*/memory.jsonl, kind 'source'); later proposals cite them "
                            f"({sum(c['judged_citing_a_source_card'] for c in counts)} judged proposals). The call is in "
                            "skeptic/streams/scout.py and runs inside the loop every 10 proposals."},
        "modal": {"status": "available; part of it is outside the repository",
                  "evidence": "runs/ledger.jsonl: 1,996 scored proposal-cell rows x 5 seeds (modal_jobs/score_stream_legacy.py); "
                              "runs/exp2: 1,440 proposal slots scored through modal_jobs/exp2.py; 40 fine-tunes on B300 GPUs "
                              "(modal_jobs/train40.py), whose caches are in the Modal volumes skeptic-open and skeptic-sealed. "
                              "Job logs and timings are in the Modal dashboard, not in the repository. The registered scores "
                              "came from a batch container with a static validator; the isolated sandbox path was smoke-tested only."},
        "pastdev": {"status": "writes and Proposer reads happened during proposal generation; not verifiable from repository artifacts",
                    "evidence": "Proposal generation on Sat 14 Nov used skeptic.memory.PastDevMemory: Scout and Proposer entries "
                                "were ingested and the Proposer recalled them before each proposal (terminal counters: 313 entries "
                                "ingested, 310 recalls, 0 errors; these counters were not saved to a file). In Experiment 2 the "
                                "Proposer read the run's local memory file, not past.dev. A remote read by a second agent was not "
                                "part of any experiment. tables/pastdev_check.json, if present, is a read-only check made afterwards."},
    }


def pastdev_check(out="tables/pastdev_check.json", memory_path="runs/memory.jsonl") -> dict:
    """Two read-only recalls against past.dev, as two named identities, saved with what came back."""
    from skeptic.agents import load_environment
    from skeptic.memory import PastDevMemory
    load_environment(Path(".env"))
    memory = PastDevMemory(memory_path)
    local = {e["id"]: e for e in memory.read()}
    record = {"made_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "note": "Read-only verification after the experiments. It is not part of any registered run.", "recalls": []}
    for reader, query in (("proposer", "source cards about Mahalanobis distance for out-of-scope detection"),
                          ("analyst", "proposals that fuse energy with a feature density")):
        hits = memory.recall(query, reader=reader)
        record["recalls"].append({"reader_identity": reader, "query": query, "hits": len(hits),
                                  "entries_returned": [{"id": i, "kind": local[i]["kind"], "written_by": local[i]["author"]}
                                                       for h in hits for i in h["ids"] if i in local][:12],
                                  "first_hit_text": hits[0]["text"][:300] if hits else ""})
    record["errors"] = memory.errors
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("export", "pastdev-check"))
    ap.add_argument("--run", default="runs/exp2/banking77_bert/B0")
    ap.add_argument("--sha", default="969fe13dd999")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.command == "pastdev-check":
        print(json.dumps(pastdev_check(args.out or "tables/pastdev_check.json"), indent=2))
        return 0
    counts = memory_arm_counts()
    summary = Path("runs/exp2/summary.json")
    bundle = {"schema_version": 1, "scope": "validation only; the sealed test set was never read",
              "generated_by": "python -m skeptic.analysis.evidence_bundle export",
              "explanations": EXPLANATIONS, "example": trace(args.run, args.sha), "memory_arm_counts": counts,
              "experiment_2_runs": json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else [],
              "three_rule_results": "tables/validation_summary.md", "sponsor_evidence": sponsor_evidence(counts)}
    out = Path(args.out or "tables/evidence_bundle.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ex = bundle["example"]
    print(f"wrote {out}: proposal {ex['proposal']['sha']} ({ex['validation']['verdict']}), {len(ex['sources'])} source card(s), "
          f"{len(ex['lessons'])} lesson(s) built from {sum(len(x['built_from_verdicts_on']) for x in ex['lessons'])} earlier verdicts, "
          f"{len(ex['later_proposals_citing_it_or_the_lesson'])} later proposal(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
