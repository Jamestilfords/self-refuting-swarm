# E1: Modal infrastructure

Owner: [@XayHanmonty](https://github.com/XayHanmonty).

## Setup from a clean clone

Prerequisites: Git, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and a Modal account. Python 3.12 is selected by `.python-version`; uv can install
it when necessary. Dependencies are pinned in `uv.lock`.

```sh
git clone https://github.com/Jamestilfords/self-refuting-swarm.git
cd self-refuting-swarm
make bootstrap
uv run --frozen modal token new  # Only if this computer is not already authenticated.
make setup
```

`make setup` installs the locked dependencies, initializes ER-1 at the pinned
submodule commit, validates any configured named secrets, creates two named
**v2 Volumes** if missing, deploys the app, and runs its two CPU mount checks.
Running it again reuses the same resource names and data. A deployment error
returns a nonzero exit code; a successful run prints JSON with the app URL,
resource IDs, and check results.

Each check writes, commits, reloads, reads, and removes its own random marker
under `.skeptic-setup/`. It checks that the other mount is absent. It does not
read dataset files or sealed test results. The checks have no GPU, no schedule,
and at most one container each; idle containers scale down after two seconds.

```sh
make setup-check  # Check the existing deployment; do not create or redeploy it.
make test         # Local tests; no network or Modal credentials required.
```

## Configuration

Set environment variables before running `make setup`:

- `MODAL_ENVIRONMENT`: existing Modal environment, default `main`.
- `MODAL_PROFILE`: optional existing Modal profile; authentication stays with the SDK.
- `SKEPTIC_APP_NAME`: app name, default `skeptic`.
- `SKEPTIC_OPEN_VOLUME`: open volume name, default `skeptic-open`.
- `SKEPTIC_SEALED_VOLUME`: sealed volume name, default `skeptic-sealed`.
- `SKEPTIC_SECRET_NAMES`: optional comma-separated existing Modal Secret names.

Open and sealed volume names must differ. Use unique names for a separate
experiment deployment. Changing names points to different resources; it does
not migrate data. An existing v1 Volume under a requested name is rejected
rather than replaced. The Modal environment must already exist.

## Named provider secrets

No provider secret is required for E1's CPU checks or Testbed A's public models.
Create provider credentials in the [Modal Secrets dashboard](https://modal.com/secrets)
when later jobs need them, then supply their names, for example:

```sh
SKEPTIC_SECRET_NAMES=skeptic-huggingface,skeptic-openai make setup
```

Setup verifies those named secrets exist. It does not upload local environment
variables, create placeholder credentials, overwrite existing secrets, or print
their values. `secrets_by_name` in `app.py` exposes lazy handles; attach only the
specific secret a trusted job needs with `secrets=[secrets_by_name[name]]`.
Secret contents and required keys must be checked by the consuming job.

Modal's token ID and token secret authenticate the local CLI/SDK. Keep them in
Modal's local profile or the standard SDK environment variables, not in a
provider secret injected into proposal execution.

## Integration for E2-E6

Import `app`, `open_volume`, `sealed_volume`, and `secrets_by_name` from
`modal_jobs.app`. Define functions in the existing job modules; setup imports
`train40`, `seal`, `sandbox`, `cache_llama`, and `score_guard` before deploying
the shared app. The app has no default volumes or secrets.

### E2 training

`train40.py` launches from the local clone and runs each fine-tune on Modal.
It fetches pinned public CLINC150 and Banking77-OOS data plus pinned BERT and
RoBERTa snapshots into `/sealed/e2/assets`. Each run uses one B300, eight CPU
cores, 32 GiB of RAM, and a two-hour timeout. Modal runs at most four training
containers concurrently. The CUDA 13.2 image and PyTorch build meet Modal's
B300 requirement of CUDA 13.1 or newer.

```sh
make bootstrap
uv run --frozen python -m modal_jobs.train40 plan
uv run --frozen python -m modal_jobs.train40 prepare
uv run --frozen python -m modal_jobs.train40 run \
  --dataset clinc150 --backbone bert --arm ce --seed 0
make train40  # all 40 runs, after the single run succeeds
```

The trusted trainer mounts both volumes. It stores model checkpoints and full
five-split caches under `/sealed/<dataset>_<backbone>/<arm>_seed<N>/`. It writes
only train logits, features, pooled vectors, and labels plus label-free `val`
and `oos_val` features to the matching `/open/` path. Test labels and test
scores stay sealed. The local command reports run status, not test metrics.
Existing runs with the same recipe are reused; a mismatched or incomplete run
fails instead of overwriting results. `prepare` and `run` need a Modal login
and network access from the trusted preparation container. The `plan` command
needs neither.

The E1 setup image and volume-check functions do not use the training preset.

After all 40 runs finish, select the Modal profile for their workspace and run
`make sanity`. It starts one CPU function with read-only access to the open
volume. The function requires all 40 cache manifests, checks that no test
arrays or validation labels were exported, and reports validation AUROC for
MSP, energy, and vacuity by run plus the mean primary score for each group of
five seeds. This check does not read the sealed volume or produce test AUROC;
paper test comparisons wait for the later unblinding workflow.

- Trusted training jobs may mount both volumes. They write full caches to
  `/sealed` and copy only `train`, `val`, and `oos_val` arrays to `/open`.
- The trusted E3 wrapper mounts `/open` read-only. Each untrusted proposal
  Sandbox receives only its prepared cell arrays and has no volume, network,
  or secrets.
- The trusted sealer mounts `/sealed`; it returns status, not test scores.

E1 establishes resource separation and verifies the setup checks. E3 enforces
the proposal execution boundary, while E4 still needs the read guard and
committed decision-hash check. A Modal workspace administrator can access both
volumes; separate volumes alone are not the complete sealed-exam enforcement.

### E3 proposal validation

With the Modal profile for the workspace containing the open caches:

```sh
make e3-plan      # Local static preview; no Modal job.
make e3-canaries  # Three valid proposals, each scored in two CPU Sandboxes.
uv run --frozen python -m modal_jobs.sandbox run \
  --proposal proposals/6cef58e34d00.py \
  --dataset clinc150 --backbone bert --arm ce --seed 0 \
  --stream canary
```

`plan` and `run` accept repeated `--proposal` flags. `--canaries` reads the
checked-in canary manifest. `--dataset`, `--backbone`, `--arm`, and `--seed`
select exactly one of the 40 E2 cells. A proposal's `ARM` must match the cell.
`--ledger PATH` selects the local append-only JSONL file, defaulting to
`runs/decisions/e3.jsonl`. Existing proposal/cell rows are reused, so a retry
does not silently replace the first validation result.
The Modal dashboard shows the trusted function in the temporary `skeptic-e3`
app (or `<SKEPTIC_APP_NAME>-e3` with an override) and the child Sandboxes under
the E1 `skeptic` app. The JSONL ledger stays on the launching machine.

The trusted CPU function verifies the E2 recipe and exact open-cache keys.
It randomly mixes in-scope and OOS validation arrays without exposing the
membership mask or shuffle seed. It runs the full batch and a hidden 10%
subset in separate Modal Sandboxes. The Sandboxes use gVisor, two CPUs, 8 GiB
RAM, a 300-second lifetime, blocked network, and no mounted volume or secret.
The worker process has a 270-second deadline for imports, input decoding,
scoring and output writing. The registered `score(ctx)` limit remains 60 seconds
per seed: increasing infrastructure time does not relax proposal validity.
The one-seed wrapper allows 12 minutes for full/subset workers and overhead;
the five-seed validation scorer allows 55 minutes. These are maximum budgets,
not runtime estimates.
The function verifies score length, finiteness, probability rows, and subset
consistency before computing validation AUROC. Only aggregate status, runtime,
and validation AUROC enter the ledger; no test array or score is read.

This is E3 execution status, not a critic verdict. The full four-canary
`make smoke` needs the R1 verdict gates and the E4 sealed guard. Running E3
requires the pinned ER-1 submodule and an existing E1 deployment in the same
workspace. It uses CPU and does not start B300 training.

SDK references: [Apps](https://modal.com/docs/guide/apps),
[Volumes](https://modal.com/docs/guide/volumes), and
[Secrets](https://modal.com/docs/guide/secrets).

## Isolated validation stream

See [run and test instructions](../docs/isolated-scoring.md). The batch stream
uses disposable sandboxes and keeps the separate trusted G5 calibration pass.
The former `score_stream --seal` route is disabled.
