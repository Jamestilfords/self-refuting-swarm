# Matched decision replay

R0, R1 and R2 replay the same proposal order in each of four cells. R0 uses
unrounded AUROCs from per-item scores. R1 uses a fixed stratified 50/50 split,
an explicit threshold, and registered seeded Laplace streams. Each cell gets
200 `numpy.random.default_rng(0).permutation` orderings over its 487 scored,
non-baseline, non-canary proposals, starting from SHAs sorted lexically. R1
draws threshold noise at the start of each ordering and again only after a
guarded answer; it draws query noise on
every scored proposal and answer noise only when the guard is used.
R2 reuses the merged `skeptic.critic_inputs` loader and critic, with fresh
LORD++ state for each cell and ordering. Guard canaries are checked separately
and never enter the registered orderings.
Registered baseline identities are excluded using the baseline manifest even
if discovery tagged a shared baseline file as a templated proposal. Canary
rows are also excluded from the experiment stream.

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
four rows per file, and one manifest. A separate test exercises all 600 files
for 200 orderings. R2 labels the small synthetic sample `UNDERPOWERED`; the
larger adapter fixture also checks `ACCEPT` and the guard canary's `BUG`
outside the proposal stream. A repeated replay is byte-identical. A frozen
output refuses further writes. These synthetic settings are not research
preregistration.

## Supply the registered inputs

Download the registered score directory while preserving
`<open-root>/scores/<group>/<sha>.npz`. Use `runs/ledger.jsonl` and
`proposals/baselines.json`. The CLI checks every score file against the
committed `runs/score_manifest.sha256` (1,996 hashes) and verifies the
ledger's AUROCs against the arrays before writing decisions.

The registered 200-ordering settings are encoded in `runs/replay-config.json`.
The JSON must explicitly contain:

- `orderings`: positive count; `ordering_seed`: nonnegative integer.
- `initial`: each of the four cell names mapped to its scored baseline SHA.
- `r1_threshold` and `r1_split_seed`.
- `r1_noise`: `distribution: laplace`, `sigma`, `seed`, and
  `algorithm: numpy-pcg64-seedsequence-v1`. Each cell and ordering gets an
  independent NumPy generator from `SeedSequence([seed, sorted-cell-index,
  ordering])`. Its draws are `Lap(2σ)` for the threshold, `Lap(4σ)` for each
  query, and `Lap(σ)` for guarded answers.
- `critic`: every field of `CriticConfig`, with approved values. Printing
  `python -m skeptic.critic.config --markdown` shows these values.
- `ignore_parents`: the three coupling SHAs declared in §3.1 of the
  preregistration. The critic does not apply G6 to them.

The tracked ledger can be planned without score files:

```sh
uv run --frozen python -m skeptic.rules.replay plan \
  --config runs/replay-config.json --ledger runs/ledger.jsonl \
  --open-root ./downloaded-open --source legacy \
  --baselines proposals/baselines.json --output .
```

Expected for the registered ledger: 487 candidates per cell, 200 orderings, and
600 decision files. No decisions are written. `plan` validates configuration
and the ledger; score manifest, artifact, and calibration checks happen during
`run`.

The preregistration is now marked `Registered` and the score manifest is
committed. Download the matching per-item score files from Modal into
`./downloaded-open/scores/`, keeping all cell folders and names. Replace
`plan` with `run`. It writes
`runs/decisions/R{0,1,2}/ord_XXX.jsonl` and
`runs/decision_manifest.json`. The manifest includes the exact configuration
and hashes of the configuration file, ledger, baseline manifest, and score
files. The runner processes one ordering at a time, caches repeated critic
bootstrap statistics, and spools complete logs to a temporary directory. The
writer checks the same completeness/chain/gate rules as the freeze guard
before changing output files. A draft registration, mismatched input, or
conflicting previous output stops the command. This is a local CPU replay; it
does not launch Modal training or scoring.

`make exp1-plan REPLAY_ARGS='...'` and `make exp1 REPLAY_ARGS='...'` are aliases
for these commands. The decision hash is a separate review step (`make
seal-hash`); this runner never freezes, commits, publishes, or unblinds results.

## Current calibrated ledger

Use the new calibrated ledger and its matching per-item score files directly;
there is no need to rerun scoring. If the files are at `runs/scores/<group>/`,
use `--open-root runs --source legacy` and point `--ledger` at the new ledger.
For isolated artifacts, select `--source isolated` instead. This preserves the
same `load_cell(cell)` -> `verdict(...)` data contract.

The tracked `runs/ledger.jsonl` is the registered 500-proposal stream (2,000
rows across four cells). Four `io` rows belong to the guard canary. After
resolving shared baseline identities, there are 487 scored replay candidates
per cell, 9 registered scores and 4 canaries. The 608-proposal stream is kept
separately as `runs/ledger_608_exploratory.jsonl` and does not enter this
replay. The matching per-item score files still need to be downloaded from
the Modal run; the ledger and checksum manifest alone are insufficient for
paired bootstrap. The CLI refuses missing files, mismatched hashes,
calibration, or AUROCs that disagree with the ledger.

`make bootstrap` installs torch. The replay CLI adds `third_party/er1/src`
automatically. For a standalone use of the existing critic-input CLI, retain
its explicit environment setting:

```sh
PYTHONPATH=third_party/er1/src uv run --frozen python -m skeptic.critic_inputs \
  --scores runs/scores --ledger runs/ledger.jsonl
```

Run that only after installing the calibrated inputs at those paths; it checks
canary verdicts locally and reports mismatches as a nonzero exit code.
