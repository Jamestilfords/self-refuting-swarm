# Matched decision replay

R0, R1 and R2 replay the same proposal order in each of four cells. R0 uses
unrounded AUROCs from per-item scores. R1 uses a fixed stratified 50/50 split,
an explicit threshold, and an externally supplied schedule of noise pairs.
R2 reuses the merged `skeptic.critic_inputs` loader and critic, with fresh
LORD++ state for each cell and ordering. Guard violations remain `BUG` without
spending a LORD++ step. Other invalid proposals are recorded as rejected.

This is offline validation replay: no proposal code, test arrays, Modal calls,
or training are involved. The PyTorch dependency is required by the critic's
pinned ER-1 imports; it does not make replay a GPU job.

## Verify locally

```sh
make bootstrap
uv run --frozen python -m pytest tests/test_replay.py tests/test_replay_cli.py tests/test_critic_bridge.py tests/test_bridge.py tests/critic -q
uv run --frozen python -m skeptic.rules.replay --help
```

Expected: tests pass. The optional real-cache baseline test may skip when its
cache fixture is unavailable. The CLI integration test builds synthetic scores
for all four cells, writes six decision files (three rules x two orderings),
eight rows per file, and one manifest. R2 labels the small synthetic sample
`UNDERPOWERED` and the blocked proposal `BUG`; the larger adapter fixture also
checks `ACCEPT`. A repeated replay is byte-identical. A frozen output refuses
further writes. These synthetic settings are not research preregistration.

## Supply the registered inputs

Download the chosen open score directory while preserving
`<open-root>/scores_isolated/<group>/<sha>.npz` (or `scores/` for legacy scores).
Use the matching local ledger and baseline manifest. Never mix score sources.
The CLI checks the score provenance and verifies the ledger's AUROCs against
those arrays before writing decisions.

The JSON configuration must explicitly contain:

- `orderings`: positive count; `ordering_seed`: nonnegative integer.
- `initial`: each of the four cell names mapped to its scored baseline SHA.
- `r1_threshold` and `r1_split_seed`.
- `r1_noise`: each cell mapped to one list per ordering. Each list has one
  `[threshold_noise, answer_noise]` pair per proposal position. Generate this
  schedule under the approved noise distribution and seed; the runner chooses
  neither for you.
- `critic`: every field of `CriticConfig`, with approved values. Printing
  `python -m skeptic.critic.config --markdown` shows proposed defaults, which
  still require the research owner's confirmation.

For example, once those real files exist:

```sh
uv run --frozen python -m skeptic.rules.replay plan \
  --config runs/replay-config.json --ledger runs/ledger_isolated.jsonl \
  --open-root ./downloaded-open --source isolated \
  --baselines proposals/baselines.json --output .
```

Expected: a JSON summary of proposal counts, orderings, and `3 * orderings`
decision files; no decisions are written. `plan` validates configuration and
the ledger; artifact and calibration checks happen during `run`.

After preregistration is complete and explicitly marked `FROZEN`, replace
`plan` with `run`. It writes `runs/decisions/R{0,1,2}/ord_XXX.jsonl` and
`runs/decision_manifest.json`. The manifest includes the exact configuration
and hashes of the configuration file, ledger, baseline manifest, and score
files. The writer checks the same completeness/chain/gate rules as the freeze
guard before changing output files. A draft registration, mismatched input,
or conflicting previous output stops the command.

`make exp1-plan REPLAY_ARGS='...'` and `make exp1 REPLAY_ARGS='...'` are aliases
for these commands. The decision hash is a separate review step (`make
seal-hash`); this runner never freezes, commits, publishes, or unblinds results.

## Teammate's calibrated 500-proposal run

Use the new calibrated ledger and its matching per-item score files directly;
there is no need to rerun scoring. If the files are at `runs/scores/<group>/`,
use `--open-root runs --source legacy` and point `--ledger` at the new ledger.
For isolated artifacts, select `--source isolated` instead. This preserves the
same `load_cell(cell)` -> `verdict(...)` data contract.

The tracked `runs/ledger.jsonl` now contains the teammate's 500-proposal run
(2,000 rows across four cells). All 1,996 successful rows have plain-model ECE;
all 1,176 successful rows that return probabilities have candidate ECE. The
per-item score files still need to be downloaded from the matching Modal run;
the ledger alone is insufficient for paired bootstrap. The CLI refuses missing
calibration or scores that disagree with the ledger.

`make bootstrap` installs torch. The replay CLI adds `third_party/er1/src`
automatically. For a standalone use of the existing critic-input CLI, retain
its explicit environment setting:

```sh
PYTHONPATH=third_party/er1/src uv run --frozen python -m skeptic.critic_inputs \
  --scores runs/scores --ledger runs/ledger.jsonl
```

Run that only after installing the calibrated inputs at those paths; it checks
canary verdicts locally and reports mismatches as a nonzero exit code.
