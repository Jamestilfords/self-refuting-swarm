# Exa and past.dev agent workflow

The local coordinator runs the existing Scout and Proposer. Scout searches Exa
for OOS scoring methods and saves source cards with URLs, queries, dates and
excerpt hashes. Proposer reads those cards and local/past.dev memory, then asks
OpenAI for a candidate scoring function. The coordinator performs static and
citation checks; generated Python executes only inside fresh Modal CPU
sandboxes with no network, volumes or secrets.

This is proposal generation and synthetic validation. It does not train models,
score the datasets, run the three statistical rules, freeze decisions or unblind
test data. Exa supplies research evidence; it does not decide keep/reject.

## Setup

Run commands from the repository root:

```sh
make agents-bootstrap
cp -n .env.example .env
chmod 600 .env
```

Fill in `.env` locally. It is ignored by Git. Use:

- `EXA_API_KEY` for research searches.
- `PASTDEV_API_KEY` for shared memory and `SKEPTIC_MEMORY_BACKEND=pastdev`.
  Use `local` to keep all memory on disk.
- `OPENAI_API_KEY` and `SKEPTIC_PROPOSER_MODEL` for the existing OpenAI Responses
  integration. Choose a model enabled for your project; the CLI does not guess one.
- Both `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`, or leave both empty and select
  a previously authenticated `MODAL_PROFILE`. A token ID starting with `ak-`
  alone is insufficient. Profile names are local names, not workspace IDs.
  The credentials/profile must belong to the intended Modal workspace
  (`jamestilfordsusanto` for the team's runs).

The agent CLI reads only `.env` in the current directory (or `--env-file PATH`).
Shell variables take precedence. It does not search parent directories,
interpolate environment values, or send the file to Modal. Other existing Modal
CLIs still use the environment/profile as documented in their guides.

```sh
make agents-plan
make agents-check
make agents-test
```

Expected results: `plan` prints the workflow and missing setting names without
contacting any service. `check` exits 0 when configuration is present, or 2 with
the missing names. It does **not** authenticate credentials. The test target uses
fake providers and authored fixtures, so it spends no service credits.

## Run a small development stream

When ready to make paid API calls and launch CPU sandboxes:

```sh
make stream AGENT_ARGS='--run-dir runs/agents/pilot --n 5'
```

Default limits are 5 total proposal attempts, 2 LLM attempts per proposal,
20 total LLM calls (including Scout summaries), 2 Exa searches, 3 search results
per search, 2,048 output tokens per LLM call, and a 150,000-token allowance.
Scout searches before proposal 0 and every 10 proposals thereafter. Override
limits with `--max-calls`, `--max-searches`, `--num-results`, `--max-output-tokens`,
`--budget-tokens`, `--scout-every` and `--max-attempts`.

OpenAI automatic retries are disabled. API calls are charged to the run budget
before dispatch. Token reservations conservatively use prompt bytes plus output
allowance and framing headroom, then settle to reported usage. Failed or
interrupted calls keep their reservation. These limits are not a dollar spending
cap. Modal checks have separate CPU charges and timeouts; each check uses up to
three fresh sandboxes for full, repeat and hidden-subset scoring.

Passing candidates have valid metadata/citations, finite nonconstant outputs,
repeatable scores/probabilities, itemwise subset agreement, and distinct rankings
from energy/MSP and earlier candidates on the synthetic context. Passing these
checks is not proof of scientific improvement or a critic ACCEPT verdict.

## Outputs and resume

Each run has its own ignored directory under `runs/agents/`:

- `session.json`: nonsecret settings and a unique memory namespace.
- `budget.json`: persisted call/search counts and token accounting.
- `memory.jsonl`: authoritative local source/proposal cards.
- `scout.json`: search cursor and skipped URLs.
- `proposals/<sha>.py` and `proposals/llm.json`: candidates and rejection reasons.
- `report.json`: attempt counts, usage, memory writers and stop reason.

Rerun the **same command** to resume. `--n` is the total target, not an additional
count. Completed targets make no further API or Modal calls. Settings and limits
must match the saved session; choose a new run directory for a different model or
budget. Prior accepted candidates are checked again in isolated sandboxes when
continuing an incomplete stream. One process may own a run directory at a time.
Do not delete state files to resume; keep the directory as one unit.

The CLI exits 0 when the requested attempts finish (some may be rejected), 2 for
missing configuration/invalid arguments, 3 when a budget stops the run, and 1 for
other failures. Review `llm.json` and `report.json`, not just the exit code.

past.dev receives only source/proposal memory cards, never model arrays or API
keys. Ingest IDs include a run namespace so separate runs do not overwrite each
other's `m_0001`. Recall drops evidence without provenance from that run and
test-related content. Local memory remains available during past.dev outages;
failed remote writes are counted and are not automatically backfilled. Retrieved
text is untrusted evidence, and provider error bodies are omitted from logs.

## Hand off candidates for dataset validation

Preview the existing E3 scorer using a separate proposal directory and ledger:

```sh
uv run --frozen python -m modal_jobs.score_stream plan \
  --proposals runs/agents/pilot/proposals --streams llm \
  --ledger runs/agents/pilot/validation-ledger.jsonl
```

Expected: a task plan for the new candidates, with no Modal execution. To score
them later, configure Modal as in [the scoring guide](isolated-scoring.md) and
replace `plan` with `run`. Per-item validation artifacts stay in Modal's open
volume. The original 500-proposal stream and its ledger stay separate. Statistical
decisions still use the registered [replay workflow](decision-replay.md), including
the required baseline score artifacts and calibration. Automatic critic feedback
and Experiment 2 orchestration are subsequent integration work.

The old `python -m skeptic.streams.proposer ...` entry point now points here and
exits without executing code. There is no production local-execution switch.
