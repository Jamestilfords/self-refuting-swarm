"""Validate the demo's real data projection and validation-only data boundary."""
import json

import pytest

from dashboard.data import ROOT, json_rows, parse_report, read_local, recorded_run, snapshot


def write_run(root, row):
    path = root / "runs/exp2/clinc150_bert/B0/decisions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n")
    return path


def test_published_summary_and_recordings():
    data = snapshot()
    assert data["validation_only"] is True
    assert data["report"]["orderings"] == 200
    cells = data["report"]["cells"]
    assert [c["candidates"] for c in cells] == [487] * 4
    assert [c["accepts"]["R0"]["mean"] for c in cells] == [6.5, 6.6, 5.4, 6.0]
    assert [c["accepts"]["R1"]["mean"] for c in cells] == [28.9, 33.0, 66.4, 37.6]
    assert all(c["critic_accepted"] == 0 for c in cells)
    assert list(data["candidate_counts"].values()) == [487] * 4
    assert len(data["runs"]) == 18
    for run in data["runs"]:
        recording = recorded_run(run["id"])
        assert len(recording["rows"]) == run["slots"] == 80
        assert sum(bool(r["accepted"]) for r in recording["rows"]) == run["accepted"]
        assert sum(r["status"] == "judged" for r in recording["rows"]) == run["judged"]
    leak = next(c for c in data["canaries"] if c["name"] == "test_leak")
    assert leak["expected"] == "BUG"
    assert leak["statuses"] == {"io": 4}


def test_default_recording_preserves_missing_interval_and_unreached_gates():
    row = recorded_run("clinc150_bert/B0")["rows"][0]
    assert row["verdict"] == "NO EFFECT"
    assert row["critic"]["ci"] is None
    assert row["critic"]["gates"] == {"G1": "pass", "G2": "pass", "G3": "fail"}


def test_only_validation_fields_are_returned(tmp_path):
    row = {"sha": "abc", "val_auroc": 0.91, "status": "judged", "source": "https://example.org/paper",
           "memory_used": ["m1"], "test_auroc": "PRIVATE", "token": "PRIVATE", "scores": ["PRIVATE"],
           "critic": {"ci": [-0.01, 0.02], "gates": {"G1": "pass", "test_truth": "PRIVATE"},
                      "test_auroc": "PRIVATE", "detail": {"test": "PRIVATE"}}}
    write_run(tmp_path, row)
    data = recorded_run("clinc150_bert/B0", tmp_path)
    assert "PRIVATE" not in json.dumps(data)
    assert data["rows"][0]["memory_count"] == 1
    assert data["rows"][0]["critic"]["ci"] == [-0.01, 0.02]
    assert data["rows"][0]["source"] == "https://example.org/paper"


@pytest.mark.parametrize("source", [None, "", "javascript:alert(1)", "file:///sealed/test.npy", "https://"])
def test_non_web_sources_and_unscored_rows(tmp_path, source):
    write_run(tmp_path, {"status": "not_scored", "source": source})
    row = recorded_run("clinc150_bert/B0", tmp_path)["rows"][0]
    assert row["source"] == ""
    assert row["val_auroc"] is None
    assert row["verdict"] is None
    assert row["critic"]["gates"] == {}


@pytest.mark.parametrize("run_id", ["../../.env", "/sealed", "clinc150_bert/B0/../../secret", "clinc150_bert/D0", "%2e%2e/.env"])
def test_run_path_is_not_user_selectable(run_id):
    with pytest.raises(ValueError, match="Unknown run"):
        recorded_run(run_id)


def test_symlink_inputs_rejected(tmp_path):
    target = tmp_path / "private.jsonl"
    target.write_text('{"secret": "private"}\n')
    link = tmp_path / "linked.jsonl"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlinks"):
        read_local(tmp_path, "linked.jsonl")
    directory = tmp_path / "redirect"
    directory.symlink_to(ROOT / "runs", target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        read_local(tmp_path, "redirect/ledger.jsonl")


def test_in_progress_jsonl_tail_is_ignored_but_corruption_is_not(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text('{"idx":0}\n{"idx":')
    assert json_rows(path) == [{"idx": 0}]
    path.write_text('{"idx":0}\n{"idx":\n')
    with pytest.raises(json.JSONDecodeError):
        json_rows(path)
    path.write_text('corrupt\n{"idx":0}\n')
    with pytest.raises(json.JSONDecodeError):
        json_rows(path)


def test_missing_or_changed_summary_fails_instead_of_inventing_results(tmp_path):
    with pytest.raises(FileNotFoundError):
        snapshot(tmp_path)
    with pytest.raises(ValueError, match="ordering count"):
        parse_report("# No results")
    report = (ROOT / "tables/validation_summary.md").read_text().replace("6.5 [2-12]", "missing")
    with pytest.raises(ValueError, match="acceptance summary"):
        parse_report(report)
