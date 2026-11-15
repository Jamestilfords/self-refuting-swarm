"""Check evidence joins, real citation counts, and degraded artifact handling."""
import hashlib
import json
import shutil

import pytest

from dashboard.data import ROOT, public_text, web_url
from dashboard.evidence import (NOTE_PATH, demo_notes, discover_runs, memory_usage,
                                proposal_source, run_health, trace_run)


def save(root, relative, value, jsonl=False):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in value) if jsonl else json.dumps(value))
    return path


def bundle():
    return {"explanations": {"research_question": "Question", "observed_result": "Observed", "limitation": "Limited"},
            "example": {"proposal": {"sha": "658fabc872e5", "slot": 60,
                                     "file": "runs/exp2/banking77_bert/B0/proposals/658fabc872e5.py"}},
            "sponsor_evidence": {"Exa": {"status": "recorded", "evidence": "Source cards"},
                                 "past.dev": {"status": "recent check", "evidence": "Separate from historical runs"}}}


def test_recorded_direct_trace_has_explicit_lesson_and_followup():
    trace = trace_run("clinc150_bert/B0")
    card = trace["cards"][3]
    assert card["sha"] == "db08a69d6d63"
    assert card["sources"]
    lesson = next(l for l in card["resulting_lessons"] if l["id"] == "m_0058")
    assert lesson["followups"][0]["slot"] == 29
    assert lesson["followups"][0]["sha"] == "f3cd2860d9e7"
    assert card["complete_trace"]
    assert "not established" in trace["memory_transport"]


def test_james_slot_60_preserves_indirect_ancestry():
    card = trace_run("banking77_bert/B0")["cards"][59]
    assert card["sha"] == "658fabc872e5"
    assert card["verdict"] == "SCORE-MATCHED"
    assert card["memory_used"] == ["m_0104"]
    assert card["sources"][0]["link"] == "Indirect memory ancestry"
    assert card["sources"][0]["via_ids"] == ["m_0104", "m_0081", "m_0068"]
    lesson = card["used_lessons"][0]
    assert not lesson["direct"]
    assert lesson["via_ids"] == ["m_0104", "m_0079", "m_0060", "m_0059"]
    assert len(lesson["built_from_verdicts_on"]) == 5
    assert all(r["slot"] < card["slot"] for r in lesson["built_from_verdicts_on"])
    assert card["complete_trace"]


def test_counts_distinguish_proposals_from_citations():
    counts = [r for r in memory_usage() if r["arm"] == "B"]
    assert len(counts) == 6 and all(r["available"] for r in counts)
    assert sum(r["source_proposals"] for r in counts) == 81
    assert sum(r["source_citations"] for r in counts) == 97
    assert sum(r["lesson_proposals"] for r in counts) == 68


def test_missing_memory_does_not_invent_a_source_or_lesson(tmp_path):
    save(tmp_path, "runs/exp2/clinc150_bert/B0/decisions.jsonl",
         [{"idx": 0, "sha": "a" * 12, "status": "judged", "memory_used": ["m_0001"]}], jsonl=True)
    trace = trace_run("clinc150_bert/B0", tmp_path)
    assert trace["warnings"]
    assert trace["cards"][0]["sources"] == []
    assert trace["cards"][0]["used_lessons"] == []
    assert not trace["cards"][0]["complete_trace"]


def test_memory_reference_cycles_are_bounded(tmp_path):
    prefix = "runs/exp2/clinc150_bert/B0/"
    save(tmp_path, prefix + "decisions.jsonl", [{"idx": 0, "sha": "a" * 12, "memory_used": ["m_0001"]}], jsonl=True)
    save(tmp_path, prefix + "memory.jsonl", [
        {"id": "m_0001", "kind": "proposal", "author": "proposer", "content": {"sha": "b" * 12, "memory_used": ["m_0002"]}},
        {"id": "m_0002", "kind": "proposal", "author": "proposer", "content": {"sha": "c" * 12, "memory_used": ["m_0001"]}},
    ], jsonl=True)
    card = trace_run("clinc150_bert/B0", tmp_path)["cards"][0]
    assert len(card["memory_ancestry"]) == 2
    assert not card["complete_trace"]


def test_source_is_hashed_and_read_as_text_without_execution(tmp_path):
    code = 'RATIONALE = "Synthetic fixture"\nraise RuntimeError("must never execute")\n'
    sha = hashlib.sha256(code.encode()).hexdigest()[:12]
    path = tmp_path / f"runs/exp2/clinc150_bert/B0/proposals/{sha}.py"
    path.parent.mkdir(parents=True)
    path.write_text(code)
    result = proposal_source("clinc150_bert/B0", sha, tmp_path)
    assert result["status"] == "Matched"
    assert result["rationale"] == "Synthetic fixture"
    path.write_text(code + "# changed\n")
    assert proposal_source("clinc150_bert/B0", sha, tmp_path)["status"] == "Hash mismatch"
    path.unlink()
    assert proposal_source("clinc150_bert/B0", sha, tmp_path)["status"] == "Missing"


def test_source_path_and_symlink_cannot_escape(tmp_path):
    with pytest.raises(ValueError):
        proposal_source("clinc150_bert/B0", "../../secret", tmp_path)
    path = tmp_path / "runs/exp2/clinc150_bert/B0/proposals/aaaaaaaaaaaa.py"
    path.parent.mkdir(parents=True)
    secret = tmp_path / "private"
    secret.write_text("PRIVATE")
    path.symlink_to(secret)
    result = proposal_source("clinc150_bert/B0", "a" * 12, tmp_path)
    assert result["status"] == "Unreadable" and not result["code"]


def test_credential_patterns_are_redacted():
    text = "Authorization: Bearer sensitivevalue123456 API_KEY=secretvalue123456 ak-abcdefghijklmnop past_sk_abcdefghijklmnop sk-abcdefghijklmnop"
    result = public_text(text)
    for value in ("sensitivevalue", "secretvalue", "abcdefghijklmnop"):
        assert value not in result
    assert web_url("https://user:password@example.com/paper") == ""
    assert web_url("https://example.com/?api_key=private") == ""


def test_bundle_uses_james_format_without_overriding_measured_fields(tmp_path):
    data = bundle()
    data["example"]["validation"] = {"test_auroc": "PRIVATE", "AUROC": 0.999999}
    data["token"] = "PRIVATE"
    save(tmp_path, NOTE_PATH, data)
    projected = demo_notes(tmp_path)
    assert projected["status"] == "Provided"
    assert projected["research"]["question"] == "Question"
    assert projected["featured_trace"] == {"run_id": "banking77_bert/B0", "proposal_sha": "658fabc872e5"}
    assert projected["sponsors"]["past.dev"]["status"] == "recent check"
    assert "PRIVATE" not in json.dumps(projected)
    assert "0.999999" not in json.dumps(projected)


@pytest.mark.parametrize("value", [[], {}, {"explanations": []}, {"explanations": {"research_question": "Only one"}}])
def test_invalid_bundle_is_reported_without_breaking_recorded_views(tmp_path, value):
    save(tmp_path, NOTE_PATH, value)
    assert demo_notes(tmp_path)["status"] == "Invalid"


def test_missing_bundle_is_optional(tmp_path):
    assert demo_notes(tmp_path)["status"] == "Not supplied"


def test_local_health_reports_real_coverage_and_missing_replay_evidence():
    health = run_health()
    checks = {c["check"]: c for c in health["checks"]}
    assert checks["Unique proposal IDs"]["observed"] == 500
    assert checks["Scoring ledger rows"]["observed"] == 2000
    assert checks["Registered run directories"]["observed"] == 18
    assert checks["Recorded registered slots"]["observed"] == 1440
    assert len(health["failures"]) == 167
    assert all(r["status"] == "Counts match" for r in health["runs"])
    assert checks["Detailed R0 replay logs"]["status"] == "Mismatch"
    assert health["freeze_verified"] is False and health["training_verified"] is False


def test_missing_artifacts_and_inconsistent_summary_get_reported(tmp_path):
    empty = run_health(tmp_path)
    assert next(c for c in empty["checks"] if c["check"] == "Recorded registered slots")["observed"] is None
    src = ROOT / "runs/exp2/banking77_bert/B0"
    dest = tmp_path / "runs/exp2/banking77_bert/B0"
    shutil.copytree(src, dest)
    save(tmp_path, "runs/exp2/summary.json", [{"run": "runs/exp2/banking77_bert/B0", "slots": 1, "judged": 1, "accepted": 1, "not_scored": 0}])
    health = run_health(tmp_path)
    run = next(r for r in health["runs"] if r["run"] == "banking77_bert/B0")
    assert "Decision counts differ from summary" in run["problems"]
    (dest / "budget.json").write_text('{"used":[],"limits":{}}')
    run = next(r for r in run_health(tmp_path)["runs"] if r["run"] == "banking77_bert/B0")
    assert "Budget unreadable" in run["problems"]


def test_additional_local_runs_are_discovered_but_not_registered(tmp_path):
    (tmp_path / "runs/exp2/clinc150_bert/B9").mkdir(parents=True)
    (tmp_path / "runs/exp2/clinc150_bert/D0").mkdir()
    assert discover_runs(tmp_path) == ["clinc150_bert/B9"]
    row = next(r for r in run_health(tmp_path)["runs"] if r["run"] == "clinc150_bert/B9")
    assert row["registered"] is False
