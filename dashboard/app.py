"""Streamlit demo: uv run --only-group dashboard streamlit run dashboard/app.py."""
from __future__ import annotations

import html
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

import altair as alt
import streamlit as st

# Streamlit adds the script directory; also allow the repository's data package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.data import label, recorded_run, snapshot  # noqa: E402

PAGES = ("Overview", "Proposal replay", "Memory study", "Guard & provenance")
ARMS = {"A": "Ratchet · own history", "B": "Critic · shared memory", "C": "Critic · no memory"}
VERDICT_COLORS = {"SCORE-MATCHED": "#749a81", "NO EFFECT": "#b8c9a5",
                  "UNDERPOWERED": "#d1ad78", "INERT": "#a598ad", "other": "#c3c9bd"}
GATES = (("G1", "Material change"), ("G2", "Score novelty"), ("G3", "Benefit evidence"),
         ("G4", "Power diagnosis"), ("G5", "Calibration"), ("G6", "Fusion ceiling"))


def esc(value):
    return html.escape(str(value if value is not None else ""), quote=True)


def number(value, digits=4, signed=False):
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def markup(content):
    st.html(content)


@st.cache_data(ttl=10, show_spinner=False)
def load_snapshot():
    return snapshot()


@st.cache_data(ttl=10, show_spinner=False)
def load_recording(run_id):
    return recorded_run(run_id)


def stop_playback():
    st.session_state.playing = False


def reset_replay():
    st.session_state.update(cursor=1, inspect_slot=1, playing=False,
                            search="", verdict_filter="All verdicts", last_tick=time.monotonic())


def toggle_playback(count):
    if st.session_state.cursor >= count:
        st.session_state.cursor = 1
        st.session_state.inspect_slot = 1
    st.session_state.playing = not st.session_state.playing
    st.session_state.last_tick = time.monotonic()


def scrub():
    stop_playback()
    st.session_state.inspect_slot = st.session_state.cursor


def open_replay(run_id="clinc150_bert/B0"):
    reset_replay()
    st.session_state.run_id = run_id
    st.session_state.page = "Proposal replay"


def advance_recording(count):
    """Advance only saved slots; this never invokes agents or scoring."""
    if not st.session_state.playing:
        return
    now = time.monotonic()
    interval = 1.5 / st.session_state.speed
    steps = int((now - st.session_state.last_tick) / interval)
    if steps:
        st.session_state.cursor = min(count, st.session_state.cursor + steps)
        st.session_state.inspect_slot = st.session_state.cursor
        st.session_state.last_tick = now
    if st.session_state.cursor >= count:
        stop_playback()


def heading(kicker, title, subtitle):
    markup(f'<div class="eyebrow">{esc(kicker)}</div>')
    st.title(title)
    markup(f'<p class="subtitle">{esc(subtitle)}</p>')


def acceptance_chart(report):
    values = [{"Cell": c["label"], "Rule": rule, "Mean": r["mean"], "Minimum": r["min"], "Maximum": r["max"]}
              for c in report["cells"] for rule, r in c["accepts"].items()]
    base = alt.Chart(alt.Data(values=values)).encode(
        y=alt.Y("Cell:N", sort=[c["label"] for c in report["cells"]], title=None,
                axis=alt.Axis(labelLimit=175, labelPadding=12)),
        yOffset=alt.YOffset("Rule:N", sort=["R0", "R1", "R2"]),
        color=alt.Color("Rule:N", scale=alt.Scale(domain=["R0", "R1", "R2"],
                                                range=["#728ba8", "#c99258", "#26765b"]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Cell:N", "Rule:N", "Mean:Q", "Minimum:Q", "Maximum:Q"],
    )
    bars = base.mark_bar(size=12, cornerRadiusEnd=3).encode(
        x=alt.X("Mean:Q", title="Accepted proposals per ordering", scale=alt.Scale(domain=[0, 100])))
    ranges = base.mark_rule(strokeWidth=2, opacity=.5).encode(x="Minimum:Q", x2="Maximum:Q")
    zero = base.transform_filter(alt.datum.Mean == 0).mark_point(filled=True, size=30).encode(x="Mean:Q")
    return (bars + ranges + zero).properties(height=350).configure_view(stroke=None).configure_axis(
        gridColor="#edf0e7", labelColor="#68765e", titleColor="#68765e", titleFontWeight="normal")


def overview(data):
    heading("EXPERIMENT 01 / REGISTERED REPLAY", "Can a better score survive scrutiny?",
            "One proposal stream. Three acceptance rules. A validation-only look at what survives.")
    report = data["report"]
    for col, title, value, help_text in zip(st.columns(4),
            ("Candidates / cell", "Random orderings", "Evaluation cells", "Closed-loop runs"),
            (report["cells"][0]["candidates"], report["orderings"], len(report["cells"]), len(data["runs"])),
            ("Baselines and canaries excluded", "Per cell; same stream, three rules", "2 datasets × 2 backbones", "Recorded; 80 proposal slots each")):
        col.metric(title, value, help=help_text, border=True)
    st.write("")
    left, right = st.columns([1.7, 1], gap="large")
    with left, st.container(border=True):
        st.subheader("How often does a rule say yes?")
        st.caption("R0 Ratchet  ·  R1 Thresholdout  ·  R2 Statistical critic")
        st.altair_chart(acceptance_chart(report), width="stretch")
        st.caption("Bars = means. Lines = minimum–maximum over 200 orderings. Hover for exact values.")
    with right, st.container(border=True):
        st.subheader("The critic’s verdict")
        selected = st.selectbox("Evaluation cell", [c["id"] for c in report["cells"]],
                                format_func=label, key="overview_cell")
        cell = next(c for c in report["cells"] if c["id"] == selected)
        markup(f'<div class="big-zero">{cell["critic_accepted"]}</div><div class="small-muted">proposals accepted across all {report["orderings"]} orderings</div>')
        total = sum(cell["verdicts"].values())
        stack = "".join(f'<span style="width:{value / total * 100:.2f}%;background:{VERDICT_COLORS[name]}"></span>'
                        for name, value in cell["verdicts"].items())
        rows = "".join(f'<div class="verdict-row"><i style="background:{VERDICT_COLORS[name]}"></i>{esc(name)}<strong>{value:.1f}%</strong></div>'
                       for name, value in cell["verdicts"].items())
        markup(f'<div class="stack">{stack}</div>{rows}<div class="callout"><b>A higher score is only the first hurdle.</b><br>Novelty, evidence of gain and the remaining checks determine acceptance.</div>')
    left, right = st.columns([1.7, 1], gap="large")
    with left, st.container(border=True):
        st.subheader("A small gain, before statistical checks")
        st.caption(f"{cell['label']} · exploratory validation AUROC, seed mean")
        a, b = st.columns(2)
        a.metric(f"Strongest baseline · {cell['best']['baseline']}", number(cell["best"]["baseline_auroc"]))
        b.metric("Best candidate", number(cell["best"]["auroc"]),
                 delta=f"{number(cell['best']['gap'], signed=True)} AUROC", delta_color="off")
        st.caption(f"Candidate {cell['best']['sha']} · this validation gain is not a confirmed test gain.")
    with right, st.container(border=True):
        st.subheader("What the evidence tells us")
        markup('<p class="insight">The critic accepted <b>zero candidates in every cell</b>. Test-side FDP, recall and H1–H4 remain uncomputed. This acceptance outcome does not establish false-discovery control.</p>')
        st.button("Explore proposal replay", icon=":material/play_arrow:", type="primary", on_click=open_replay)
    with st.expander("Inspect the published acceptance numbers"):
        st.dataframe([{"Cell": c["label"], **{r: f"{v['mean']:.1f} [{v['min']:.0f}–{v['max']:.0f}]"
                       for r, v in c["accepts"].items()}} for c in report["cells"]], hide_index=True)


def proposal_detail(row, slot):
    verdict = row["verdict"] or "NOT SCORED"
    tone = "" if row["accepted"] else "rejected"
    markup(f'<div class="proposal-head"><div><div class="code-id">SLOT {slot:02d} / {esc(row["sha"] or "UNSCORED")}</div><h3>{esc(row["family"] or "Unscored proposal")}</h3></div><span class="tag {tone}">{esc(verdict)}</span></div>')
    if row["status"] != "judged":
        st.info("This slot did not produce a scored proposal. No validation result or critic verdict is available.")
        return
    markup(f'<p class="rationale">{esc(row["summary"] or "No decision explanation recorded.")}</p>')
    a, b = st.columns(2)
    a.metric("Validation AUROC", number(row["val_auroc"]))
    b.metric("Gain vs incumbent", number(row["delta_inc"], signed=True))
    st.caption(f"Incumbent: {row['incumbent']} · recorded seed means")
    critic = row["critic"]
    if row["rule"] == "critic":
        markup('<div class="eyebrow">CRITIC GATES / RECORDED OUTCOMES</div>')
        chips = []
        for key, name in GATES:
            status = critic["gates"].get(key)
            tone = status if status in ("pass", "fail") else "diagnosis" if status else ""
            chips.append(f'<div class="gate {tone}"><b>{key} · {esc(status or "Not reached")}</b>{name}</div>')
        markup(f'<div class="gates">{"".join(chips)}</div>')
        if critic["ci"]:
            st.write(f"**95% gain interval:** {number(critic['ci'][0], signed=True)} to {number(critic['ci'][1], signed=True)}")
        else:
            st.caption("95% gain interval: not computed in this decision.")
        st.caption(f"Comparator: {critic['comparator'] or 'not reached'} · gain {number(critic['delta_val'], signed=True)}")
        st.caption(f"One-sided p: {number(critic['p'], 6)} · LORD++ α: {number(critic['alpha_t'], 6)}")
    else:
        st.info("Arm A uses a keep-if-better ratchet. Critic gates do not apply to this arm.")
    st.caption(f"Memory: {row['memory_count']} recalled entries · scoring status: {row['status']}")
    if row["source"]:
        st.link_button("Read research source", row["source"], icon=":material/open_in_new:")
    else:
        st.caption("No research source recorded.")


def replay(data):
    heading("EXPERIMENT 02 / RECORDED WALKTHROUGH", "Follow a proposal through the critic.",
            "Inspect the evidence and the first gate that stops a candidate. Playback uses saved decisions.")
    markup('<span class="tag recorded">RECORDED PLAYBACK · NO AGENTS OR MODAL JOBS LAUNCHED</span>')
    runs = {r["id"]: r for r in data["runs"]}
    if not runs:
        st.info("No recorded runs are available.")
        return
    if st.session_state.run_id not in runs:
        st.session_state.run_id = next(iter(runs))
        reset_replay()
    st.selectbox("Recorded run", list(runs), key="run_id", on_change=reset_replay,
                 format_func=lambda rid: f"{label(runs[rid]['cell'])} / {rid.split('/')[-1]} / {ARMS[runs[rid]['arm']]}")
    try:
        recording = load_recording(st.session_state.run_id)
    except (ValueError, KeyError, TypeError, OSError):
        stop_playback()
        st.error("This recording is unavailable or invalid. Select another run or restore its local decision file.")
        return
    rows = recording["rows"]
    if not rows:
        stop_playback()
        st.info("This recording has no complete decision rows yet.")
        return
    st.session_state.cursor = max(1, min(st.session_state.cursor, len(rows)))
    advance_recording(len(rows))
    play_col, reset_col, speed_col, note_col = st.columns([1.1, .8, .8, 2.5], vertical_alignment="bottom")
    play_col.button("Pause" if st.session_state.playing else "Play recording", key="play", type="primary",
                    icon=":material/pause:" if st.session_state.playing else ":material/play_arrow:",
                    on_click=toggle_playback, args=(len(rows),), width="stretch")
    reset_col.button("Reset", key="reset", icon=":material/replay:", on_click=reset_replay, width="stretch")
    speed_col.selectbox("Speed", [1, 2, 4], format_func=lambda n: f"{n}×", key="speed")
    note_col.caption("Screen playback only. Validation files refresh every 10 seconds.")
    if len(rows) > 1:
        st.slider("Recorded proposal slot", min_value=1, max_value=len(rows), key="cursor", on_change=scrub)
    cursor = st.session_state.cursor
    accepted = sum(bool(r["accepted"]) for r in rows[:cursor])
    st.progress(cursor / len(rows), text=f"Slot {cursor} of {len(rows)} · {accepted} accepted so far" + (" · recording complete" if cursor == len(rows) else ""))
    left, right = st.columns([1, 1.25], gap="large")
    with left, st.container(border=True):
        st.subheader("Proposal stream")
        st.caption(ARMS[runs[st.session_state.run_id]["arm"]])
        st.text_input("Search family or proposal ID", key="search", placeholder="Search the revealed proposals…")
        st.selectbox("Verdict", ["All verdicts", "SCORE-MATCHED", "NO EFFECT", "UNDERPOWERED", "INERT", "ACCEPT", "KEPT", "DISCARDED", "NOT SCORED"], key="verdict_filter")
        visible = [i for i in range(1, cursor + 1)
                   if st.session_state.search.lower() in f"{rows[i-1]['family']} {rows[i-1]['sha']} {rows[i-1]['summary']}".lower()
                   and (st.session_state.verdict_filter == "All verdicts"
                        or (rows[i-1]["verdict"] or "NOT SCORED") == st.session_state.verdict_filter)]
        if visible:
            if st.session_state.inspect_slot not in visible:
                st.session_state.inspect_slot = visible[-1]
            st.selectbox("Inspect proposal", list(reversed(visible)), key="inspect_slot", on_change=stop_playback,
                         format_func=lambda i: f"{i:02d} · {rows[i-1]['family'] or 'unscored'} · {rows[i-1]['verdict'] or 'NOT SCORED'}")
            st.dataframe([{"Slot": i, "Family": rows[i-1]["family"] or "unscored", "Verdict": rows[i-1]["verdict"] or "NOT SCORED",
                           "AUROC": rows[i-1]["val_auroc"]} for i in reversed(visible)], hide_index=True, height=280,
                          column_config={"AUROC": st.column_config.NumberColumn(format="%.4f")})
            st.caption(f"{len(visible)} matching proposals shown")
        else:
            st.info("No revealed proposals match. Adjust the filter or advance the recording.")
    with right, st.container(border=True):
        if visible:
            proposal_detail(rows[st.session_state.inspect_slot - 1], st.session_state.inspect_slot)
        else:
            st.caption("Select a matching proposal to inspect its evidence.")


def memory_study(data):
    heading("EXPERIMENT 02 / MEMORY ABLATION", "Does shared memory reduce repetition?",
            "Three arms, three recorded runs per benchmark. Compare the registered wasted-proposal measure.")
    cell = st.selectbox("Benchmark", ["banking77_oos/bert", "clinc150/bert"], format_func=label, key="memory_cell")
    runs = [r for r in data["runs"] if r["cell"] == cell]
    for col, arm in zip(st.columns(3), ARMS):
        selected = [r for r in runs if r["arm"] == arm]
        with col, st.container(border=True):
            st.subheader(f"Arm {arm}")
            st.caption(ARMS[arm])
            st.caption(f"{len(selected)} runs · {sum(r['judged'] for r in selected)} judged proposals")
            for key, text, color in (("wasted_first_half", "Slots 1–40", "#abc1a0"),
                                      ("wasted_second_half", "Slots 41–80", "#56825f")):
                values = [r[key] for r in selected if r[key] is not None]
                value = mean(values) * 100 if values else None
                markup(f'<div class="study-row"><div class="study-label">{text}<b>{number(value, 1)}%</b></div><div class="study-bar"><span style="width:{max(0, min(100, value or 0)):.2f}%;background:{color}"></span></div></div>')
            changes = [r["wasted_change"] for r in selected if r["wasted_change"] is not None]
            st.metric("Mean change in wasted share", f"{number(mean(changes) * 100 if changes else None, 1, signed=True)} pp")
    with st.container(border=True):
        st.subheader("The measure saturates in every arm")
        markup('<p class="insight">A judged proposal is “wasted” when it is a <b>rejection whose (family, verdict) pair appeared among earlier rejections</b>. With few families and verdicts, nearly every pair has appeared after 40 slots. The published late-run shares reach <b>94–100%</b>. H5 is descriptive; this does not establish a benefit from shared memory.</p>')
    with st.container(border=True):
        st.subheader("Explore the recorded runs")
        st.dataframe([{"Run": r["id"].split("/")[-1], "Judged": r["judged"], "Slots": r["slots"], "Accepted": r["accepted"],
                       "First 40 (%)": round(r["wasted_first_half"] * 100, 1) if r["wasted_first_half"] is not None else None,
                       "Last 40 (%)": round(r["wasted_second_half"] * 100, 1) if r["wasted_second_half"] is not None else None} for r in runs], hide_index=True)
        if runs:
            selected = st.selectbox("Run to inspect", [r["id"] for r in runs], key="memory_run")
            st.button("Open this recording", on_click=open_replay, args=(selected,), type="primary", icon=":material/play_arrow:")


def provenance(data):
    heading("GUARDRAILS / PROVENANCE", "Keep the evidence open. The test set sealed.",
            "Every displayed result has a source. This console reads only approved validation inputs.")
    for col, title, text in zip(st.columns(4), ("01 / Propose", "02 / Validate", "03 / Replay", "04 / Keep sealed"),
            ("Recorded IDs, families and research sources.", "Saved validation scores and guard outcomes.",
             "Rule comparisons and recorded critic gates.", "Test truth, FDP and recall remain unavailable.")):
        with col, st.container(border=True):
            st.subheader(title)
            st.caption(text)
    left, right = st.columns([1.4, 1], gap="large")
    with left, st.container(border=True):
        st.subheader("Canary expectations & observations")
        st.caption("Expected critic verdicts are distinct from scoring-ledger statuses.")
        for c in data["canaries"]:
            statuses = " · ".join(f"{esc(status)} × {n} cells" for status, n in c["statuses"].items()) or "No rows"
            markup(f'<div class="canary"><div class="canary-title"><b>{esc(c["name"].replace("_", " ").title())}</b><span class="tag neutral">Expected: {esc(c["expected"])}</span></div><p>{esc(c["why"])}</p><div class="small-muted">Observed in scoring ledger: {statuses}</div></div>')
        st.caption('“ok” means scoring succeeded, not a confirmed critic verdict.')
    with right, st.container(border=True):
        st.subheader("Data provenance")
        st.write("The overview uses James’s published validation summary. It is not recalculated from the large Experiment 1 decision logs.")
        for path in data["sources"] + ["runs/exp2/{benchmark}/{arm}{seed}/decisions.jsonl"]:
            markup(f'<code class="source-path">{esc(path)}</code>')
        st.caption(f"Overview fingerprint: {data['revision'][:16]}")
        st.caption(f"Latest input modification: {datetime.fromisoformat(data['updated_at']).strftime('%Y-%m-%d %H:%M UTC')}. File time, not an experiment timestamp.")
        markup('<div class="callout"><b>Read-only application.</b><br>No training, agent, scoring, unblinding or write controls. The data adapter never opens .env, sealed stores or test arrays. Only selected validation fields are displayed.</div>')


def main():
    st.set_page_config(page_title="SKEPTIC · Research console", page_icon="≠", layout="wide")
    markup(f'<style>{Path(__file__).with_name("style.css").read_text()}</style>')
    defaults = {"page": "Overview", "run_id": "clinc150_bert/B0", "cursor": 1,
                "inspect_slot": 1, "playing": False, "speed": 1, "search": "",
                "verdict_filter": "All verdicts", "last_tick": time.monotonic()}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    with st.sidebar:
        markup('<div class="brand"><div class="brand-mark">≠</div><div><div class="brand-name">SKEPTIC</div><div class="brand-sub">RESEARCH CONSOLE</div></div></div>')
        st.caption("WORKSPACE")
        st.radio("Navigation", PAGES, key="page", label_visibility="collapsed", on_change=stop_playback)
        markup('<div class="sidebar-note"><b>● Validation workspace</b><br>Explore the evidence.<br>Keep the test set sealed.<br><br><small>EXPERIMENTS 01 / 02</small></div>')
        st.caption("Self-refuting swarm · local demo")
        st.caption("Recorded playback · live file refresh")

    active_page = st.session_state.page

    @st.fragment(run_every=.5 if active_page == "Proposal replay" else 10)
    def console():
        # A button inside this fragment can navigate to another view. Rebuild the
        # whole app so the sidebar selection and refresh interval follow it.
        if st.session_state.page != active_page:
            st.rerun()
        top, action = st.columns([6, 1], vertical_alignment="center")
        with top:
            markup(f'<div class="topline"><span>Workspace / <strong>{esc(st.session_state.page)}</strong></span><span class="tag">VALIDATION ONLY · TEST SET SEALED</span></div>')
        if action.button("Refresh", key="refresh", icon=":material/refresh:", help="Refresh the local validation inputs"):
            load_snapshot.clear()
            load_recording.clear()
        try:
            data = load_snapshot()
            st.session_state.last_snapshot = data
        except (ValueError, KeyError, TypeError, OSError):
            st.warning("Local inputs could not be refreshed. Showing the last loaded data, if available. Restore the input files and use Refresh.")
            data = st.session_state.get("last_snapshot")
        if not data:
            st.error("No validation snapshot is available in this checkout.")
            return
        {"Overview": overview, "Proposal replay": replay, "Memory study": memory_study,
         "Guard & provenance": provenance}[st.session_state.page](data)
        markup(f'<div class="footer">SKEPTIC / Evidence before acceptance · local inputs refresh every 10s · snapshot {esc(data["revision"][:8])}</div>')

    console()


if __name__ == "__main__":
    main()
