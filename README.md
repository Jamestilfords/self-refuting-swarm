# Skeptic

Does autoresearch fool itself? Skeptic compares a keep-if-better ratchet,
Thresholdout, and a statistical critic on the same proposal stream, measuring
false discoveries and recall. Testbed A uses ER-1; Testbed B extends the critic
to probe deferral on a safety monitor.

This repository follows slide 22 of **Skeptic: Design Document and Agent
Handoff** (November 14, 2026). **E1 is implemented:** `make setup` provisions
the Modal app and separate open/sealed volumes, validates any configured named
secrets, and checks both mounts. The experiment pipeline remains a scaffold.

## Getting started

### 1. Install prerequisites and clone

You need Git, Make, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and a [Modal account](https://modal.com/). Use a shell on macOS, Linux, or WSL.
The project selects Python 3.12 via `.python-version`; uv can install it if it
is not already available. You do not need a local GPU.

```sh
git clone https://github.com/Jamestilfords/self-refuting-swarm.git
cd self-refuting-swarm
uv sync --frozen
```

This creates `.venv/` and installs the versions recorded in `uv.lock`, including
the Modal SDK. You can run all commands through uv without activating the venv.

### 2. Sign in to Modal and choose the workspace

On a computer that is not already authenticated, run:

```sh
uv run --frozen modal token new
```

Complete the browser sign-in, then inspect the configured profiles:

```sh
uv run --frozen modal profile list
uv run --frozen modal profile current
```

To share experiment caches, teammates must use profiles with access to the same
Modal workspace and select the same environment and resource names. GitHub
repository access does not grant Modal workspace access. A personal workspace
creates a separate deployment and separate volumes.

The commands below use the active profile and the `main` Modal environment.
To use another existing profile, export `MODAL_PROFILE` with its name before
running them. The environment can be selected with `MODAL_ENVIRONMENT` and
must already exist in that workspace. Modal's environment named `main` is
independent of the Git branch named `main`.

### 3. Provision E1

```sh
export MODAL_ENVIRONMENT=main
make setup
```

Setup installs locked dependencies, initializes the pinned ER-1 submodule,
creates or reuses the `skeptic-open` and `skeptic-sealed` v2 volumes, deploys
the `skeptic` app, and runs short CPU storage checks. Each check mounts only
its own volume and writes, reads, and removes a temporary setup marker.
The open volume is intended for training/validation data; the sealed volume
is reserved for full caches and test scores in later work packages.

A successful run exits with code 0 and prints JSON containing `"status": "ok"`,
a `dashboard_url`, two distinct volume IDs, and `"other_mount_absent": true`
for both checks. Open the printed URL to see the app in Modal.

Rerunning setup with the same configuration reuses the existing volumes and
their data. It deploys the code in your checkout, so use the agreed team
revision when deploying to shared resources. E1 starts no training jobs;
the proposal sandbox and sealed-test guard still belong to E3 and E4.

### 4. Verify the deployment or run local tests

```sh
make setup-check  # Check the existing app and mounts without redeploying.
make test         # Run local infrastructure tests; no Modal credentials needed.
make help         # Show implemented and pending commands.
```

`setup-check` performs the same temporary-marker checks and requires a Modal
login. For local development only, `uv sync --frozen` followed by `make test`
is sufficient; you do not need to provision Modal resources.

### 5. Configure optional provider secrets

E1's checks require only the Modal login. When later jobs need Hugging Face,
OpenAI, or other credentials, create named secrets in the selected workspace
and environment using the [Modal Secrets dashboard](https://modal.com/secrets).
Then export their names, for example:

```sh
export SKEPTIC_SECRET_NAMES="skeptic-huggingface,skeptic-openai"
make setup
```

This example assumes those two secrets already exist. Setup validates their
names; consuming jobs must validate the keys they require. Provider credentials
are separate from the Modal token ID and token secret used to authenticate your
local profile. Keep credential values in Modal, not in repository files.
Setup reads exported environment variables; it does not load a `.env` file.

Resource-name overrides are `SKEPTIC_APP_NAME`, `SKEPTIC_OPEN_VOLUME`, and
`SKEPTIC_SEALED_VOLUME`. Open and sealed names must differ. For isolated work,
give all three resources distinct names to avoid redeploying the shared app.
See [Modal setup and job integration](modal_jobs/README.md) for configuration
details and the interfaces future job modules should use.

### Troubleshooting

- **`uv` or `make` is not found:** install the missing prerequisite and reopen
  your shell so it is on `PATH`.
- **Modal authentication fails:** run `uv run --frozen modal token new`, or
  select an existing authenticated profile with `MODAL_PROFILE`.
- **A configured resource is not found:** confirm the workspace profile,
  `MODAL_ENVIRONMENT`, and any `SKEPTIC_SECRET_NAMES`. For the initial app and
  volume creation, use `make setup` before `make setup-check`.
- **An existing volume has the wrong version:** E1 requires v2. Choose new
  volume names for a separate setup; existing data is not migrated automatically.
- **`make train40` or another experiment target says `Not implemented`:** that
  work package is still pending. The working E1 commands are `bootstrap`,
  `setup`, `setup-check`, and `test`.

## Repository layout

```text
self-refuting-swarm/
├── PREREGISTRATION.md
├── README.md
├── Makefile
├── pyproject.toml
├── uv.lock
├── third_party/
│   └── er1/                   # Pinned MIT-licensed git submodule
├── modal_jobs/
│   ├── app.py                 # Shared Modal app and isolated mount checks
│   ├── config.py              # Resource names, no credential values
│   ├── setup.py               # E1 provision/check entrypoint
│   ├── train40.py
│   ├── seal.py
│   ├── sandbox.py
│   ├── cache_llama.py
│   └── score_guard.py
├── skeptic/
│   ├── api.py
│   ├── sandbox.py
│   ├── store.py
│   ├── ledger.py
│   ├── memory.py
│   ├── critic/
│   │   ├── g1.py ... g6.py
│   │   ├── lord.py
│   │   └── verdict.py
│   ├── rules/
│   │   ├── ratchet.py
│   │   ├── thresholdout.py
│   │   └── critic.py
│   ├── streams/
│   │   ├── templated.py
│   │   ├── proposer.py
│   │   ├── scout.py
│   │   ├── canaries.py
│   │   └── analyst.py
│   ├── analysis/
│   │   ├── truth.py
│   │   ├── exp1.py
│   │   ├── exp2.py
│   │   └── scorecard.py
│   └── testbed_b/
│       ├── data.py
│       ├── probe.py
│       ├── band.py
│       ├── signals.py
│       └── exchange.py
├── proposals/
├── runs/
│   └── decisions/
├── tables/
├── figures/
└── dashboard/
```

Python package directories also contain `__init__.py`. The existing root-level
`cache_acts.py` and `lens_features.py` precede this scaffold; the new pipeline
does not call them. Use the package paths above for new design-doc components,
including `skeptic/sandbox.py`, `skeptic/store.py`, and `skeptic/ledger.py`.

## Work ownership

Work owners are the repository's GitHub collaborators:

- **[@mid9t](https://github.com/mid9t) (Ryu / PI):** preregistration,
  `skeptic/critic/`, truth and analysis, Testbed B
  science specification, unblinding, and findings.
- **[@Jamestilfords](https://github.com/Jamestilfords) (research engineer):**
  `skeptic/api.py`, `skeptic/streams/`,
  `skeptic/memory.py`, `proposals/`, and Experiment 2 (`skeptic/analysis/exp2.py`).
- **[@XayHanmonty](https://github.com/XayHanmonty) (software engineer):**
  `modal_jobs/`, `skeptic/sandbox.py`,
  `skeptic/store.py`, `skeptic/ledger.py`, `skeptic/rules/`, `dashboard/`,
  and the Makefile/reproduction workflow.

The proposal API, ledger/decision rows, and verdict function are the shared
interfaces described on slides 8-11. These files are placeholders, so their
presence does not mean those contracts have been implemented or validated.

## Commands

Run `make help` for the full command list from slide 22 and the associated work
packages. `setup`, `setup-check`, `bootstrap`, and `test` are implemented.
`train40`, `sanity`, `smoke`, `stream`, `exp1`, `exp2`, `seal-hash`, `unblind`,
`report`, `reproduce`, and `testbed-b` remain reserved targets that fail with an
implementation-pending message. Setup runs small CPU storage checks; it does
not start training or read dataset/test scores.

## ER-1 dependency and credit

[ER-1 (Epistemic Ranker), by mid9t](https://github.com/mid9t/ER-1-Epistemic-Ranker-)
is an MIT-licensed dependency, pinned as a git submodule at
`e877e9159d6c099ffdc06ac54d32aafc977aa437` (`e877e9159d` in the design doc).
Its source remains in the submodule with its original license and attribution.
To populate it after cloning:

```sh
git submodule update --init third_party/er1
```

The design doc's organizer approval question about prior work remains a team
decision. This scaffold does not establish that approval. No code from the
Activation-Monitoring repository has been added by this scaffold.

## Research state

`PREREGISTRATION.md` is a draft template, not a frozen registration. @mid9t must
confirm the proposed defaults and record the preregistration commit before a
main run. Keep sealed test arrays and scores outside git, the public ledger,
shared memory, and agent inputs. E4 must implement and verify the guard and
decision-hash checks before any unblinding workflow is enabled.
