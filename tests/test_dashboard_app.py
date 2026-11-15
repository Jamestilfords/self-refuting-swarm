"""Exercise Streamlit widgets against the committed validation recordings."""
import time
import tomllib
from pathlib import Path

import pytest

st = pytest.importorskip("streamlit", reason="Install the optional dashboard group to test the UI")
from streamlit.testing.v1 import AppTest  # noqa: E402

import dashboard.data as data  # noqa: E402
import dashboard.evidence as evidence  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "dashboard/app.py"


@pytest.fixture(autouse=True)
def clear_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def app(page="Overview"):
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    if page != "Overview":
        at.radio(key="page").set_value(page).run()
    assert not at.exception
    return at


def rendered_html(at):
    return "\n".join(element.proto.body for element in at.get("html"))


def test_overview_and_cell_selection():
    at = app()
    assert [m.value for m in at.metric[:4]] == ["487", "200", "4", "18"]
    at.selectbox(key="overview_cell").set_value("clinc150/bert").run()
    assert not at.exception
    assert at.metric[-1].value == "0.9835"
    assert "17.0%" in rendered_html(at)
    assert "does not establish false-discovery control" in rendered_html(at)


def test_replay_scrubbing_intervals_unscored_slots_and_filters():
    at = app("Proposal replay")
    assert at.selectbox(key="run_id").value == "clinc150_bert/B0"
    assert "NO EFFECT" in rendered_html(at)
    assert "G3 · fail" in rendered_html(at)
    assert "G4 · Not reached" in rendered_html(at)
    assert any("not computed" in c.value for c in at.caption)
    at.slider(key="cursor").set_value(13).run()
    assert any("did not produce a scored proposal" in i.value for i in at.info)
    assert not at.metric
    at.slider(key="cursor").set_value(16).run()
    assert "UNDERPOWERED" in rendered_html(at)
    assert "G4 · INCONCLUSIVE" in rendered_html(at)
    assert any("-0.0003 to +0.0011" in m.value for m in at.markdown)
    at.slider(key="cursor").set_value(80).run()
    at.selectbox(key="verdict_filter").set_value("NO EFFECT").run()
    assert set(at.dataframe[0].value["Verdict"]) == {"NO EFFECT"}
    at.text_input(key="search").set_value("no-such-proposal").run()
    assert any("No revealed proposals match" in i.value for i in at.info)
    at.button(key="reset").click().run()
    assert not at.exception
    assert at.slider(key="cursor").value == 1
    assert at.text_input(key="search").value == ""
    assert at.selectbox(key="verdict_filter").value == "All verdicts"


def test_play_pause_speed_and_arm_switch():
    at = app("Proposal replay")
    at.button(key="play").click().run()
    assert at.session_state["playing"]
    at.session_state["last_tick"] = time.monotonic() - 3.1
    at.run()
    assert at.slider(key="cursor").value >= 3
    at.button(key="play").click().run()
    assert not at.session_state["playing"]
    at.selectbox(key="speed").set_value(4).run()
    assert at.session_state["speed"] == 4
    at.button(key="play").click().run()
    at.selectbox(key="run_id").set_value("banking77_bert/A0").run()
    assert not at.session_state["playing"]
    assert at.slider(key="cursor").value == 1
    assert any("Arm A uses a keep-if-better" in i.value for i in at.info)
    assert "CRITIC GATES / RECORDED OUTCOMES" not in rendered_html(at)
    assert not at.exception


def test_memory_navigation_to_replay_and_provenance():
    at = app("Memory study")
    assert len(at.metric) == 6
    assert [m.value for m in at.metric[-3:]] == ["81", "97", "68"]
    assert len(at.dataframe[0].value) == 6  # Direct memory usage across the six Arm B runs.
    assert len(at.dataframe[1].value) == 9  # Three arms × three seeds for the selected cell.
    at.selectbox(key="memory_cell").set_value("clinc150/bert").run()
    at.selectbox(key="memory_run").set_value("clinc150_bert/C2").run()
    next(b for b in at.button if b.label == "Open this recording").click().run()
    assert at.radio(key="page").value == "Proposal replay"
    assert at.selectbox(key="run_id").value == "clinc150_bert/C2"
    at.radio(key="page").set_value("Guard & provenance").run()
    assert not at.exception
    assert "Expected: BUG" in rendered_html(at)
    assert "io × 4 cells" in rendered_html(at)
    assert any("not a confirmed critic verdict" in c.value for c in at.caption)


def test_missing_snapshot_retains_last_data_with_warning(monkeypatch):
    at = app()

    def unavailable():
        raise FileNotFoundError("private path should not be displayed")

    monkeypatch.setattr(data, "snapshot", unavailable)
    at.button(key="refresh").click().run()
    assert not at.exception
    assert len(at.warning) == 1
    assert at.metric[0].value == "487"
    assert "private path" not in at.warning[0].value
    empty = AppTest.from_file(str(APP), default_timeout=30).run()
    assert any("No validation snapshot" in e.value for e in empty.error)
    assert not empty.exception


def test_missing_recording_stops_playback(monkeypatch):
    at = app("Proposal replay")
    at.button(key="play").click().run()

    def unavailable(_run_id):
        raise FileNotFoundError("private recording path")

    monkeypatch.setattr(data, "recorded_run", unavailable)
    at.button(key="refresh").click().run()
    assert not at.exception
    assert not at.session_state["playing"]
    assert any("recording is unavailable" in e.value for e in at.error)
    assert not at.metric


def test_proposal_text_is_escaped(monkeypatch):
    original = data.recorded_run

    def recording(run_id):
        result = original(run_id)
        result["rows"][0]["summary"] = '<img src=x onerror="alert(1)">'
        return result

    monkeypatch.setattr(data, "recorded_run", recording)
    at = app("Proposal replay")
    assert "&lt;img src=x" in rendered_html(at)
    assert "<img src=x" not in rendered_html(at)
    assert not at.exception


def test_local_server_defaults():
    config = tomllib.loads((APP.parents[1] / ".streamlit/config.toml").read_text())
    assert config["server"]["address"] == "127.0.0.1"
    assert config["server"]["enableStaticServing"] is False
    assert config["browser"]["gatherUsageStats"] is False


def test_james_trace_shows_ancestry_code_and_validation():
    at = app("Research trace")
    # Keep this ancestry example independent of the optional bundle's featured selection.
    at.selectbox(key="trace_run_id").set_value("banking77_bert/B0").run()
    at.selectbox(key="trace_slot").set_value(60).run()
    assert at.selectbox(key="trace_run_id").value == "banking77_bert/B0"
    assert at.selectbox(key="trace_slot").value == 60
    assert "658fabc872e5" in rendered_html(at)
    assert "SCORE-MATCHED" in rendered_html(at)
    assert at.metric[-1].value == "+0.0024"
    assert at.code and "def score(ctx)" in at.code[0].value
    assert any("hash matches" in s.value for s in at.success)
    assert "Lesson found through earlier proposal references" in rendered_html(at)
    assert any("m_0104 → m_0079 → m_0060 → m_0059" in c.value for c in at.caption)
    assert any("indirect link does not establish" in i.value for i in at.info)
    assert at.get("download_button")
    next(b for b in at.button if b.label == "Inspect this slot in playback").click().run()
    assert at.radio(key="page").value == "Proposal replay"
    assert at.selectbox(key="run_id").value == "banking77_bert/B0"
    assert at.slider(key="cursor").value == 60
    assert not at.exception


def test_trace_follows_direct_lesson_to_later_proposal():
    at = app("Research trace")
    at.selectbox(key="trace_run_id").set_value("clinc150_bert/B0").run()
    at.selectbox(key="trace_slot").set_value(4).run()
    assert at.selectbox(key="trace_followup").value == 29
    at.button(key="open_followup").click().run()
    assert at.selectbox(key="trace_slot").value == 29
    assert "Directly declared lesson: m_0058" in rendered_html(at)
    assert not at.exception


def test_trace_filters_handle_runs_without_complete_chains():
    at = app("Research trace")
    at.selectbox(key="trace_run_id").set_value("banking77_bert/A0").run()
    at.checkbox(key="complete_only").check().run()
    assert any("no complete source-to-lesson-to-follow-up" in i.value for i in at.info)
    at.checkbox(key="complete_only").uncheck().run()
    at.selectbox(key="trace_slot").set_value(13).run()
    assert any("Recorded reason: duplicate" in c.value for c in at.caption)
    assert not at.exception


def test_health_exposes_missing_logs_and_failure_navigation():
    at = app("Run health")
    assert [m.value for m in at.metric] == ["18 / 18", "1440", "167", "0 / 600"]
    assert any("not verified" in i.value for i in at.info)
    at.button(key="inspect_failure").click().run()
    assert at.radio(key="page").value == "Research trace"
    assert at.selectbox(key="trace_slot").value == 13
    assert any("Recorded reason" in c.value for c in at.caption)
    assert not at.exception


def test_health_works_even_when_the_overview_snapshot_is_missing(monkeypatch):
    def unavailable():
        raise FileNotFoundError("private path")

    monkeypatch.setattr(data, "snapshot", unavailable)
    at = app("Run health")
    assert at.metric[0].value == "18 / 18"
    assert not at.error and not at.exception


def test_missing_trace_is_visible_without_private_error_details(monkeypatch):
    def unavailable(_run):
        raise FileNotFoundError("private path")

    monkeypatch.setattr(evidence, "trace_run", unavailable)
    at = app("Research trace")
    assert any("decision log is missing or invalid" in e.value for e in at.error)
    assert "private path" not in at.error[0].value
    assert not at.exception


def test_curated_notes_are_editorial_and_refreshable(monkeypatch):
    def notes():
        return {"status": "Provided", "prepared_by": "Research team",
                "research": {"question": "Curated question", "observed_result": "Curated observation", "limitation": "Curated limitation"},
                "featured_trace": {"run_id": "clinc150_bert/B0", "proposal_sha": "db08a69d6d63"},
                "sponsors": {"past.dev": {"status": "Recent check", "evidence": "Two named recall checks"}}}

    monkeypatch.setattr(evidence, "demo_notes", notes)
    at = app()
    assert "Curated question" in rendered_html(at)
    assert at.metric[0].value == "487"
    at.button(key="overview_trace").click().run()
    assert at.selectbox(key="trace_run_id").value == "clinc150_bert/B0"
    assert at.selectbox(key="trace_slot").value == 4
    at.radio(key="page").set_value("Guard & provenance").run()
    assert "Two named recall checks" in rendered_html(at)
    assert any("later past.dev recall check does not prove" in c.value for c in at.caption)
    assert not at.exception
