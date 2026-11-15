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

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git,
then from a clone of this repository run:

```sh
uv sync --frozen
uv run modal token new  # Skip if this computer already has a Modal login.
make setup
```

Setup also initializes the pinned ER-1 submodule. `make setup-check` verifies
the existing deployment; `make test` runs local infrastructure tests without
Modal credentials. See [Modal setup and job integration](modal_jobs/README.md)
for resource names, named-secret configuration, and mount boundaries.

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
