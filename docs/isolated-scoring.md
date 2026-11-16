# Isolated validation scoring

The stream scorer now runs each proposal in fresh gVisor sandboxes with no
volumes, network, secrets, or Modal credentials. The trusted wrapper has the
open volume, checks each of five seeds plus a hidden subset, restores item
order, and writes per-item scores. The separate calibration function runs no
proposal code and reads only validation labels through `SealedStore`.

The ledger still has the existing calibration fields (`ece`, `ece_plain`) and
`has_probs`, so the merged critic bridge can consume it. Isolated artifacts go
to `/open/scores_isolated/<group>/<sha>.npz`; the local full ledger is
`runs/ledger_isolated.jsonl`. A filtered run writes `ledger_isolated_partial.jsonl`.
Older scores and ledgers are not overwritten. Use one source consistently.

Banking77 jobs use the training identifier `banking77_oos` for run validation
and cache-recipe checks; the on-volume directory remains `banking77_<backbone>`.
This applies to both BERT and RoBERTa and avoids the earlier recipe mismatch.

Infrastructure timeouts allow 300 seconds per sandbox and 270 seconds for the
worker process, including imports and array decoding. The five-seed scorer has
a 55-minute job timeout. The registered 60-second `score(ctx)` limit is unchanged;
a function exceeding that limit still receives `timeout`, even if the sandbox
has time remaining. These limits are execution ceilings, not runtime estimates.

## Local verification (no Modal jobs)

```sh
make bootstrap
uv run --frozen python -m pytest tests/test_score_stream.py tests/test_runner.py tests/test_e3.py tests/test_store.py -q
uv run --frozen python -m modal_jobs.score_stream plan --streams canary,baseline
```

Expected: the tests pass; `plan` prints the number of proposal/cell jobs and
exits without calling Modal. Tests use synthetic arrays and mocked cloud calls.
They check restored item ordering, probability outputs, safe retries, source
provenance, calibration in the ledger, and rejection of the legacy test route.

## Run on Modal when ready

Use the Modal profile for the workspace holding the 40 trained caches. This
launches CPU scoring jobs and uses credits; it does not train models.

```sh
uv run --frozen modal profile current
uv run --frozen python -m modal_jobs.score_stream run --streams canary,baseline
```

Expected: successful proposals have five validation AUROCs; probability
proposals also have five ECE values. The leak canary has `status: io`.
`sealed` is `not_run` for successful rows. These are execution statuses, not
critic verdicts. Infrastructure failures return a nonzero exit code; incomplete
calibration stops the run before a new ledger is written. A retry reuses only
artifacts whose schema and proposal provenance validate.

The old `--seal` flag is rejected, and `runner.score_test`/`sealed_run` raise
`GuardError` before reading any test arrays. Guarded test scoring is delivered
separately. Full test metrics never enter this validation ledger.
