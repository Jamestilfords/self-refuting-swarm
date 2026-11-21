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

## Run the Streamlit dashboard

The dashboard uses the validation results and recorded Experiment 2 runs already
in this repository. You need Git, Make, and
[uv](https://docs.astral.sh/uv/getting-started/installation/); no GPU, Modal login,
API keys, or training run is required.

From an existing checkout on `main`:

```sh
git pull --ff-only origin main
make dashboard
```

For a new checkout:

```sh
git clone https://github.com/Jamestilfords/self-refuting-swarm.git
cd self-refuting-swarm
make dashboard
```

Open [http://localhost:8501](http://localhost:8501). The command installs the locked
Streamlit dashboard dependencies automatically; `make bootstrap` is not needed
for the dashboard. Stop the server with **Ctrl-C**.

Use the sidebar to explore **Overview**, **Research trace**, **Proposal replay**,
**Memory study**, **Run health**, and **Guard & provenance**. The overview shows 487 candidates per cell, 200
orderings, four evaluation cells, and 18 recorded closed-loop runs. In Proposal
replay, press **Play recording**, scrub the timeline, or filter proposals to
inspect their scores, intervals, and critic gates. Research trace connects the
Exa sources, original code, verdicts and direct/indirect memory references. Run
health shows missing artifacts, failure reasons and recorded token budgets.

Local inputs refresh every 10 seconds. Playback uses saved decisions and starts
no agents or paid jobs. The app displays validation data only; test-side FDP and
recall remain unavailable.

To test the dashboard or choose another port:

```sh
make dashboard-test  # Expected: all dashboard tests pass.
make dashboard DASHBOARD_ARGS='--server.port 8502'
```

For the second command, open [http://localhost:8502](http://localhost:8502).
If `make dashboard` reports "No rule to make target", confirm you are in the
repository root on the latest `main`. If the port is already in use, stop the
other local server or choose another port as above. See the
[dashboard demo guide](dashboard/README.md) for a two-minute research walkthrough,
expected verdicts, data sources, and limitations.

## Getting started with training and experiments

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

The sealer supports offline batch previews and bounded batch execution:

```sh
uv run --frozen python -m modal_jobs.seal plan --all
```

The preview requires `runs/decision_manifest.json` and the matching proposal
sources, but starts no Modal jobs and reads no test data. After the reviewed
freeze is published, `run --all` scores the manifest's proposal/cell jobs in
chunks, records status-only progress, and safely reuses matching sealed scores.
See the [batch and analysis handoff notes](docs/modal-sealer.md#handoff-to-the-truthreport-implementation)
for reference-baseline authorization and the remaining reporting integration.

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

2026-11-19: `make report` computes test truth in parallel and caches each pair under `runs/truth/`. Pairs, seeds, B and every table are unchanged; equivalence is tested with 1 vs 2 workers.

2026-11-20. The decision loader reads acceptance from `verdict` for R0 and R1 rows, maps the CE-energy hash `b1e4ba2e46d5` to `energy_ce`, and ignores the per-gate `detail` field. The published decision files are unchanged. These changes make the published format readable; they do not change any registered definition.

2026-11-20. The report reads validation scores in their published `.npz` format, checked against `runs/score_manifest.sha256`. No value changes.

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

## Experiment 1 results

Confirmatory orderings are 0–99. The support rules are the registered ones in `PREREGISTRATION.md` §1. Tables are in `tables/`. Strict FDP was not computed: test truth was built with `strict=False`, so `true_strict` is empty and `fdp_strict_mean` is blank in every row of `tables/exp1_fdp_recall.csv`.

### Hypotheses

| ID | Estimate | 95% CI | Holm p | Verdict under the registered rule |
|---|---|---|---|---|
| H1 | 0.000952 | [0.000313, 0.001710] | 0.009 | Supported. The interval is above 0 and the Holm p-value is at most 0.05. |
| H2 | — | — | — | Not evaluable. R2 accepted 0 changes in all 400 cell-orderings, so every paired FDP difference was excluded. Support would have required the interval above 0 in at least 3 of the 4 cells and a Holm p-value at most 0.05. |
| H3 | 0.000000 | [0.000000, 0.000000] | 0.002 | Not supported. Support requires the point estimate to be at least 0.5. The estimate is 0. The interval and the Holm p-value are reported either way; the Holm test rejects a ratio of 0.5 and does not by itself support the claim. |
| H4 | 0.066168 | [−0.066197, 0.202020] | 0.724 | Not supported. The interval is not entirely above 0, and the Holm p-value is above 0.05. |

### FDP and recall

Pooled stream, mean over the orderings where the value is defined. `dropped` is the number of orderings, out of 100, in which that rule accepted nothing, so FDP is undefined. Recall is the share of S_cell, the proposals whose test gain over CE energy has a combined 95% interval above 0. The templated and llm splits are in `tables/exp1_fdp_recall.csv`. On CLINC150/BERT the templated slice of S_cell is empty, so that slice's recall is undefined.

| Rule | Cell | FDP | Recall | Dropped | \|S_cell\| |
|---|---|---|---|---|---|
| R0 | Banking77-OOS/BERT | 0.606 [0.575, 0.636] | 0.022 [0.021, 0.024] | 0 | 267 |
| R0 | Banking77-OOS/RoBERTa | 0.673 [0.645, 0.699] | 0.021 [0.020, 0.023] | 0 | 310 |
| R0 | CLINC150/BERT | 0.850 [0.825, 0.876] | 0.015 [0.000, 0.035] | 0 | 2 |
| R0 | CLINC150/RoBERTa | 0.562 [0.533, 0.590] | 0.030 [0.025, 0.034] | 0 | 46 |
| R1 | Banking77-OOS/BERT | 0.734 [0.720, 0.747] | 0.099 [0.091, 0.107] | 0 | 267 |
| R1 | Banking77-OOS/RoBERTa | 0.769 [0.758, 0.779] | 0.105 [0.099, 0.111] | 0 | 310 |
| R1 | CLINC150/BERT | 0.665 [0.658, 0.673] | 0.250 [0.190, 0.310] | 0 | 2 |
| R1 | CLINC150/RoBERTa | 0.650 [0.640, 0.660] | 0.219 [0.206, 0.232] | 0 | 46 |
| R2 | Banking77-OOS/BERT | undefined | 0.000 [0.000, 0.000] | 100 | 267 |
| R2 | Banking77-OOS/RoBERTa | undefined | 0.000 [0.000, 0.000] | 100 | 310 |
| R2 | CLINC150/BERT | undefined | 0.000 [0.000, 0.000] | 100 | 2 |
| R2 | CLINC150/RoBERTa | undefined | 0.000 [0.000, 0.000] | 100 | 46 |

R0's templated slice of CLINC150/RoBERTa drops 8 orderings. Every other R0 and R1 cell-stream in the confirmatory log drops none, except where a stream's S_cell is empty and recall is left blank. R2, R2 without G1, R2 without G5, and R2 without G6 drop all 100 orderings in every cell.

### Gate ablation

The contrast is the mean, over orderings, of false accepts under the variant minus false accepts under full R2. Full R2 accepted nothing, so its FDP is undefined and `d_fdp_vs_R2` is blank. Intervals are the two-level bootstrap on the false-accept contrast.

| Variant | Cell | Extra false accepts | 95% CI | Slipped per ordering |
|---|---|---|---|---|
| R2 without G1, G5, or G6 | all four | 0.00 | [0.00, 0.00] | 0.00 |
| R2 without G2 | CLINC150/BERT | 0.00 | [0.00, 0.00] | 0.00 |
| R2 without G2 | CLINC150/RoBERTa | 0.00 | [0.00, 0.00] | 0.00 |
| R2 without G2 | Banking77-OOS/BERT | 0.17 | [0.04, 0.39] | 0.17 |
| R2 without G2 | Banking77-OOS/RoBERTa | 0.05 | [0.00, 0.14] | 0.05 |
| R2 without G3 | CLINC150/BERT | 3.04 | [1.38, 5.29] | 3.04 |
| R2 without G3 | CLINC150/RoBERTa | 1.26 | [0.25, 2.64] | 1.26 |
| R2 without G3 | Banking77-OOS/BERT | 1.16 | [0.27, 2.43] | 1.16 |
| R2 without G3 | Banking77-OOS/RoBERTa | 2.03 | [0.32, 4.68] | 2.03 |

### Final pick

The last accepted proposal of each ordering, or CE energy when the ordering accepted nothing. The comparison density is the best of `d_port_ce`, `d_full_ce`, `d_port_edl`, and `d_full_edl` by validation AUROC. The AUROC below is the seed-mean test AUROC. R0 and R2 have IQR 0: every ordering ends on the same score. R2 ends on CE energy in every cell.

| Rule | Cell | Median test AUROC | CE energy | Best validation density | That density's test AUROC |
|---|---|---|---|---|---|
| R0 | Banking77-OOS/BERT | 0.941052 | 0.926771 | d_port_ce | 0.941450 |
| R0 | Banking77-OOS/RoBERTa | 0.940589 | 0.927752 | d_port_ce | 0.944068 |
| R0 | CLINC150/BERT | 0.976140 | 0.979219 | d_full_ce | 0.977679 |
| R0 | CLINC150/RoBERTa | 0.983125 | 0.981002 | d_full_ce | 0.982679 |
| R2 | Banking77-OOS/BERT | 0.926771 | 0.926771 | d_port_ce | 0.941450 |
| R2 | Banking77-OOS/RoBERTa | 0.927752 | 0.927752 | d_port_ce | 0.944068 |
| R2 | CLINC150/BERT | 0.979219 | 0.979219 | d_full_ce | 0.977679 |
| R2 | CLINC150/RoBERTa | 0.981002 | 0.981002 | d_full_ce | 0.982679 |

R1's medians move between orderings. On Banking77-OOS/BERT the median is 0.941052 (IQR 0.001390), against CE energy 0.926771 and `d_port_ce` 0.941450. On Banking77-OOS/RoBERTa it is 0.941791 (IQR 0.002940), against 0.927752 and 0.944068. On CLINC150/BERT it is 0.978906 (IQR 0.001021), against 0.979219 and `d_full_ce` 0.977679. On CLINC150/RoBERTa it is 0.982423 (IQR 0.000990), against 0.981002 and 0.982679.

### Scorecard

Eight published templated pairs, graded on validation only. A strict match is the registered verdict. A lenient match includes the registered near misses, and a strict match counts as lenient. Result: 7 strict matches out of 8, and 7 lenient matches out of 8.

| # | Scorecard verdict | Stream verdict | Grade |
|---|---|---|---|
| 1 | SCORE-MATCHED | SCORE-MATCHED | strict |
| 2 | INERT | INERT | strict |
| 3 | NO EFFECT | NO EFFECT | strict |
| 4 | NO EFFECT | NO EFFECT | miss |
| 5 | ACCEPT | SCORE-MATCHED | strict |
| 6 | ACCEPT | SCORE-MATCHED | strict |
| 7 | ACCEPT | SCORE-MATCHED | strict |
| 8 | NO EFFECT | NO EFFECT | strict |

Pair 2 is EDL D_port without whitening, proposal `c0fdb8397000`. Pair 4's validation gain is about −0.00009, so the critic returns NO EFFECT. The strict expectation was INERT, and SCORE-MATCHED was the only lenient alternative, so the pair is a miss.

## Research state

`PREREGISTRATION.md` is registered; its status line records commit
`5b0e6b223c60a3bab3ddfd058e12caa818330a7c`. The 500-proposal ledger and
the score-file manifest are committed. The matching per-item score files must
be downloaded before the main replay. The critic's `g2_mode="scorecard"` is the
scorecard rule in §4.6; the proposal stream stays on `g2_mode="stream"`.
Keep sealed test arrays and scores outside git, the public ledger, shared
memory, and agent inputs. E4 supplies the guard and decision-hash checks.
The confirmatory Experiment 1 tables are in `tables/`, and the registered
reading of them is in the results section above.

### E5 decision replay

See the [decision replay guide](docs/decision-replay.md) for setup, local tests,
the 200-ordering configuration, and expected outputs. Existing calibrated
ledgers and their matching per-item scores can be replayed without rerunning
scoring. This does not automatically freeze or unblind the experiment.
