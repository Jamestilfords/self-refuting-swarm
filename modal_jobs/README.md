# E1: Modal infrastructure

Owner: [@XayHanmonty](https://github.com/XayHanmonty).

## Setup from a clean clone

Prerequisites: Git, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and a Modal account. Python 3.12 is selected by `.python-version`; uv can install
it when necessary. Dependencies are pinned in `uv.lock`.

```sh
git clone https://github.com/Jamestilfords/self-refuting-swarm.git
cd self-refuting-swarm
uv sync --frozen
uv run modal token new  # Only if this computer is not already authenticated.
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

- Trusted training jobs may mount both volumes. They write full caches to
  `/sealed` and copy only `train`, `val`, and `oos_val` arrays to `/open`.
- Experiment/proposal execution mounts only `/open`, preferably using
  `open_volume.with_mount_options(read_only=True)`, and gets no provider or
  Modal account credentials.
- The trusted sealer mounts `/sealed`; it returns status, not test scores.

E1 establishes resource separation and verifies the setup checks. E3 still
needs its proposal sandbox and E4 still needs the read guard and committed
decision-hash check. A Modal workspace administrator can access both volumes;
separate volumes alone are not the complete sealed-exam enforcement.

SDK references: [Apps](https://modal.com/docs/guide/apps),
[Volumes](https://modal.com/docs/guide/volumes), and
[Secrets](https://modal.com/docs/guide/secrets).
