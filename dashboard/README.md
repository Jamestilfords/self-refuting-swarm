# SKEPTIC research console (E7)

A local Streamlit dashboard for the committed validation results and recorded
Experiment 2 decisions. Interactive Altair charts, proposal inspection, playback,
search, verdict filters, memory comparisons, and source provenance are implemented
in Python. No API keys, Modal login, training, scoring, or unblinding are needed.

## Start

From an existing checkout on `main`, with
[uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
git pull --ff-only origin main
make dashboard
```

Open **http://localhost:8501**. The command installs only the locked `dashboard`
dependency group and runs Streamlit. It does not install the agent SDKs or
training dependencies; `make bootstrap` is not required. For a new clone, run
`git clone https://github.com/Jamestilfords/self-refuting-swarm.git`, then
`cd self-refuting-swarm` and `make dashboard`. Stop with Ctrl-C. Use a different
port if needed:

```sh
make dashboard DASHBOARD_ARGS='--server.port 8502'
```

Then open **http://localhost:8502**. If the server reports the port is in use,
stop the other local server or choose an unused port. If Make cannot find the
target, confirm you are in the repository root on the latest `main`.

The equivalent command is:

```sh
uv run --frozen --only-group dashboard streamlit run dashboard/app.py
```

`.streamlit/config.toml` supplies the theme, binds to `127.0.0.1`, disables
usage telemetry, and keeps static file serving off. Share the browser window
for the demo; this is a local application, not a public hosted service.

## A two-minute research demo

1. **0:00–0:15 — Overview:** state the research question and show the three-rule
   comparison. Keep the validation-only limitation visible.
2. **0:15–1:10 — Research trace:** open `banking77_bert/B0`, slot 60. Follow its
   source query and method, rationale, hash-matched code and `SCORE-MATCHED`
   verdict. Expand the memory reference paths and cited earlier verdicts. The
   source and lesson links here are indirect; do not describe them as direct reads.
3. **1:10–1:35 — Memory study:** show 81 proposals citing source cards, 97 total
   source citations, and 68 proposals citing lessons across the six Arm B runs.
   Show the saturation limitation alongside these usage counts.
4. **1:35–2:00 — Run health:** show the 18 runs and 1,440 slots, the 167 unscored
   slots and their reasons, and the missing detailed replay/freeze artifacts.
   End with the research limitations and reproducible run commands.

For a direct lesson-to-follow-up example, use Research trace → `clinc150_bert/B0`
→ slot 4 → `m_0058` → slot 29. The trace view also offers a validation-only JSON
export for offline review. See [the evidence handoff](EVIDENCE_CONTRACT.md).

## Explore all six views

1. **Overview:** compare R0 ratchet, R1 Thresholdout, and R2 critic across four
   dataset/backbone cells. Bars show mean acceptances and lines show minimum–
   maximum over 200 orderings; hover for exact values. Change the evaluation
   cell to inspect critic verdict shares and the best-candidate comparison.
2. **Proposal replay:** the default is `clinc150_bert/B0`. Press **Play recording**
   or scrub the timeline. Inspect a proposal to see its saved explanation,
   source, validation AUROC, gain, interval when computed, and actual critic
   gate outcomes. Search and filter the revealed proposals. Speed, pause and
   reset are available. An absent gate is **Not reached**. Arm A uses a ratchet,
   so critic gates do not apply. Unscored slots have no invented results.
3. **Memory study:** compare arms A/B/C on both BERT benchmarks and open a run
   in replay. The wasted share is the fraction of judged proposals that are
   rejections whose `(family, verdict)` pair appeared among earlier rejections
   in the same run. Cards average the three runs equally. H5 is descriptive;
   the measure saturates and does not establish a shared-memory benefit.
4. **Research trace:** inspect Exa source cards, original rationale and code,
   actual gate outcomes, cited earlier verdicts, and direct/indirect memory
   ancestry. Use the complete-trace filter or follow a linked later proposal.
   Source files are hashed and displayed as text, never executed.
5. **Run health:** compare registered counts with the current local files,
   inspect unscored reasons and recorded budgets, and export a local health
   report. Additional local run directories are labeled separately. This view
   remains available if the overview summary is missing.
6. **Guard & provenance:** show canary expectations alongside observed
   scoring-ledger statuses and source files. The leak canary has `io` in four
   cells; `BUG` is its expected critic verdict, not a verdict established by
   those ledger rows. `ok` means scoring succeeded, not critic acceptance.
   Provider evidence from James’s bundle is attributed to that source; a recent
   past.dev recall check does not establish historical experiment retrieval.

## Expected results and limits

With the committed inputs from main at `7fb5363`:

- Four cells, 487 candidates per cell, 200 orderings and three rules.
- R0 means: 6.5, 6.6, 5.4, 6.0 accepts per ordering.
- R1 means: 28.9, 33.0, 66.4, 37.6; R2: zero in every cell.
- Eighteen recorded Experiment 2 runs, 80 proposal slots each.
- The default replay begins with a fusion proposal, `NO EFFECT`, passing G1
  and G2 and failing G3. Its interval was not computed.
- Slot 16 of the default replay is `UNDERPOWERED`, with a reported interval
  about `[-0.0003, +0.0011]` and G4 `INCONCLUSIVE`.
- FDP, recall, winner's-curse gap and H1–H4 remain uncomputed. Zero critic
  acceptances do not establish false-discovery control.

This is **recorded playback with live local-file refresh**, not a live agent or
Modal job monitor. Streamlit fragments refresh the data every 10 seconds; the
replay screen ticks every half-second for playback, using a 10-second data cache.
Use **Refresh** to clear the cache immediately. The overview displays James's
published summary and does not recompute it when the ledger changes. A missing
snapshot shows a warning and retains the last loaded snapshot if one exists;
an unavailable recording stops playback and displays an error.

## Inputs and access boundary

The data adapter reads only:

- `tables/validation_summary.md` — published, rounded Experiment 1 results.
- `runs/exp2/summary.json` — recorded run totals and descriptive H5 metrics.
- `runs/exp2/{banking77,clinc150}_bert/{A,B,C}{0,1,2}/decisions.jsonl` — a
  selected recording; an allowlist projects only its validation fields.
- Each discovered validation run’s `memory.jsonl`, `proposals/llm.json`,
  `proposals/<sha>.py`, and `budget.json` — projected trace and usage fields.
- Optional `tables/evidence_bundle.json` — James’s editorial notes and featured
  proposal selector. The adapter follows the [published format](EVIDENCE_CONTRACT.md).
- File metadata for `tables/pastdev_check.json`, `runs/decision_manifest.json`,
  `runs/DECISIONS.sha256`, `runs/guard.log`, `runs/score_manifest.sha256`, run
  state files, and the 600 expected detailed replay logs. Presence is not
  integrity verification; raw recall and guard payloads are not displayed.
- `runs/ledger.jsonl`, `proposals/baselines.json`, `proposals/canaries.json` —
  candidate counts and scoring statuses. Raw ledger rows are never displayed.

The application also reads its own CSS. It does not load `.env`, sealed stores,
raw test arrays, arbitrary user-selected paths, or the large Experiment 1
replay logs. Input symlinks are rejected. Text is escaped before rendering as
HTML, and research links are restricted to HTTP(S). There are no write, launch,
scoring, or unblinding controls. The dashboard is intended for trusted local use.

## Verify

```sh
make dashboard-test
```

Expected result: all dashboard data, evidence and UI tests pass.

This uses only the `dashboard` and `dev` groups. Data tests check the published
numbers, all 18 recordings, field projection, missing and partial files, and
path/symlink rejection. Streamlit AppTest exercises navigation, cell selection,
playback state, scrubbing, filtering, searching, missing intervals, unscored
slots, Arm A, memory study, provenance, research ancestry, portable evidence,
run-health checks, bundle integration, and missing-input messages.

For visual verification, open the app, visit all six views, resize the window,
and follow the demo above. The UI tests exercise Streamlit widgets and state;
they do not replace a rendered-browser layout inspection.

Owned by [@XayHanMonty](https://github.com/XayHanMonty).
