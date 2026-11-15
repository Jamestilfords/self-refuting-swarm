# Skeptic

Does autoresearch fool itself? Skeptic compares a keep-if-better ratchet,
Thresholdout, and a statistical critic on the same proposal stream, measuring
false discoveries and recall. Testbed A uses ER-1; Testbed B extends the critic
to probe deferral on a safety monitor.

This repository follows slide 22 of **Skeptic: Design Document and Agent
Handoff** (November 14, 2026). E1 provisions separate Modal volumes, E2 runs
training and audits the open caches, and E3 executes validation proposals in
CPU Sandboxes with an append-only ledger. The validation critic and E4 sealed-store
freeze checks and matched decision replay are implemented. The registered run
and unblinding still require their approved inputs and analysis.

## Getting started

### 1. Install prerequisites and clone

You need Git, Make, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and a [Modal account](https://modal.com/). Use a shell on macOS, Linux, or WSL.
The project selects Python 3.12 via `.python-version`; uv can install it if it
is not already available. You do not need a local GPU.

```sh
git clone https://github.com/Jamestilfords/self-refuting-swarm.git
cd self-refuting-swarm
make bootstrap
```

This creates `.venv/`, installs the versions recorded in `uv.lock`, and checks
out the pinned ER-1 submodule. You can run all commands through uv without
activating the venv.

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
make test         # Run local project tests; no Modal credentials needed.
make help         # Show implemented and pending commands.
```

`setup-check` performs the same temporary-marker checks and requires a Modal
login. For local development only, `make bootstrap` followed by `make test`
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
- **An experiment target says `Not implemented`:** that work package is still
  pending. See `make help` for the implemented E1-E3 commands.

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
│   │   ├── config.py
│   │   ├── bootstrap.py
│   │   ├── sources.py
│   │   ├── decisions.py
│   │   ├── truth.py
│   │   ├── ablation.py
│   │   ├── hypotheses.py
│   │   ├── report_tables.py
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

The proposal API, validation critic, E3 validation ledger, and E4 sealed-store
freeze checks and matched R0/R1/R2 replay are implemented. Registered settings
and complete scoring/unblinding integration remain separate work.

## Commands

Run `make help` for the full command list from slide 22 and the associated work
packages. `setup`, `setup-check`, `bootstrap`, `test`, `train40`, `sanity`,
`e3-plan`, `e3-canaries`, `seal-hash`, `seal-check`, `exp1-plan`, and `exp1` are
implemented. The bounded `stream` agent workflow and `agents-bootstrap`,
`agents-plan`, `agents-check`, and `agents-test` are also available; see
[the Exa/past.dev agent guide](docs/agents.md) for private `.env` setup, run limits,
Modal isolation, output files and resume behavior. `smoke`, `exp2`, `unblind`,
`report`, `reproduce`, and `testbed-b` remain reserved targets. Setup runs small
CPU storage checks; it does not start training or read dataset/test scores.

To preview the E2 matrix, run `uv run --frozen python -m modal_jobs.train40 plan`.
After all 40 runs finish, run `make sanity` with the Modal profile for the
workspace containing their caches. It audits the open volume and reports
validation AUROC without reading test data. See [the training guide](modal_jobs/README.md#e2-training).

### E3 validation runner

Select a Modal profile with access to the workspace containing the E2 open
caches. `make e3-plan` checks the four planted proposals locally without
starting a job. `make e3-canaries` runs the three valid CE canaries on
`clinc150_bert/ce_seed0` in Modal CPU Sandboxes; the leak canary is rejected
by the static checker before any Sandbox starts. No B300 training is launched.

```sh
make e3-plan
make e3-canaries
uv run --frozen python -m modal_jobs.sandbox run \
  --proposal proposals/6cef58e34d00.py \
  --dataset clinc150 --backbone bert --arm ce --seed 0 \
  --stream canary
```

The local append-only summary is `runs/decisions/e3.jsonl`; rerunning the same
proposal and cell reuses its row. Use `--ledger PATH` for a separate run. Rows
contain status, validation AUROC, runtime, and provenance, with no raw scores
or test fields. The trusted Modal function reads one verified open cache and
sends anonymous evaluation arrays to two fresh, network-blocked Sandboxes with
no mounted volume or secrets. One Sandbox scores the full batch and the other
checks a hidden 10% subset. The CLI prints the same public summary.

E3's `ok` status means the proposal passed validation and produced a score;
it is **not** a critic acceptance verdict. The full `make smoke` target awaits
experiment-runner integration. See [Modal E3 details](modal_jobs/README.md#e3-proposal-validation).

### Isolated validation stream

See the [isolated scoring guide](docs/isolated-scoring.md) for local tests, a
no-compute plan command, and the Modal scoring command with expected outputs.
The legacy test-scoring route is disabled.

### E4 sealed store and freeze checks

See the [sealed-store and freeze guide](docs/sealed-store.md) for the decision
file contract, `seal-hash` and `seal-check` commands, publication verification,
and trusted-job API. These tools do not start scoring or unblind a dataset.
The guarded Modal test scorer is available separately from validation. See
[sealer setup, tests and expected results](docs/modal-sealer.md); a published
experiment freeze is required before it reads test arrays.

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

## Experiment 1 truth and metrics

`skeptic/analysis/` is the pre-registered answer key for Experiment 1. It reads decision logs and, after unsealing, scores accepted changes on test data. It does not re-run the ratchet, Thresholdout, or the critic. `python -m skeptic.analysis.config --markdown` prints the definitions for `PREREGISTRATION.md` §4. Run it with `PYTHONPATH=third_party/er1/src` when you import the scoring code; the config table itself does not need ER-1.

`paired_bootstrap_fast` is an equivalence-tested vectorisation of ER-1's `paired_bootstrap` at commit `e877e9159d`. It draws the same resample indices and returns the same tie-aware weighted AUROC. Truth intervals use B = 5,000. The first truth pair in a process is checked against ER-1 at B = 20.

The filled draft records these choices. They freeze only when that file is registered:

- **D1.** Recall's numerator is the accepted proposals that sit in S_cell, the set of real improvements over CE energy. The numerator and the denominator use that same set, so recall cannot exceed 1.
- **D2.** Intervals use `paired_bootstrap_fast`, bit-identical to ER-1, with B = 5,000.
- **D3.** The two-level bootstrap reweights accepted changes. It does not replay the rules.
- **D4.** All four hypothesis p-values are two-sided: `p = max(2·min(P*(stat ≤ c), P*(stat ≥ c)), 1/B)`, with c = 0 for H1, H2 and H4 and c = 0.5 for H3. H2's p-value is the third-smallest per-cell p-value. H1, H2 and H4 are supported only when their interval rule holds and the Holm-adjusted p-value is at most 0.05. H3 is supported when the point estimate is at least 0.5.
- **D5.** Final-pick ECE uses the proposal's in-scope test probabilities when those were stored, otherwise the plain-model ECE of its arm. The best single density is chosen by validation AUROC.
- **D6.** Strict truth skips a cell whose scores are missing and does not call that a reversal. A reversal is a combined 95% interval lying entirely below 0.
- **D7.** Truth is the gain over the incumbent at acceptance, even when the critic compared the proposal with a baseline rather than that incumbent.
- **D8.** An ordering where either the ratchet or the critic accepted nothing is left out of H2's paired difference, and the count of those orderings is reported.

A finite test set misses some real gains. In the low-power synthetic check (`delta_auc = 0.005`) no planted improvement cleared a 95% interval above zero, so S_cell was empty and measured recall was undefined rather than written as 0. When that recall is defined, the test requires it to be at most the recall against the planted effects.

## Research state

`PREREGISTRATION.md` is registered; its status line records commit
`5b0e6b223c60a3bab3ddfd058e12caa818330a7c`. The 500-proposal ledger and
the score-file manifest are committed. The matching per-item score files must
be downloaded before the main replay. The critic's `g2_mode="scorecard"` is the
scorecard rule in §4.6; the proposal stream stays on `g2_mode="stream"`.
Keep sealed test arrays and scores outside git, the public ledger, shared
memory, and agent inputs. E4 supplies the guard and decision-hash checks;
scoring integration and the approved unblinding analysis remain pending.

### E5 decision replay

See the [decision replay guide](docs/decision-replay.md) for setup, local tests,
the 200-ordering configuration, and expected outputs. Existing calibrated
ledgers and their matching per-item scores can be replayed without rerunning
scoring. This does not automatically freeze or unblind the experiment.
