# Skeptic

Does autoresearch fool itself? Skeptic compares a keep-if-better ratchet,
Thresholdout, and a statistical critic on the same proposal stream, measuring
false discoveries and recall. Testbed A uses ER-1; Testbed B extends the critic
to probe deferral on a safety monitor.

This repository is the scaffold from slide 22 of **Skeptic: Design Document and
Agent Handoff** (November 14, 2026). The Python files describe their assigned
components; the experiment pipeline and Modal infrastructure are not implemented
yet. `make help` lists the planned commands. Other Makefile targets fail with an
explicit implementation-pending message.

## Repository layout

```text
self-refuting-swarm/
├── PREREGISTRATION.md
├── README.md
├── Makefile
├── third_party/
│   └── er1/                   # Pinned MIT-licensed git submodule
├── modal_jobs/
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

- **Ryu / PI:** preregistration, `skeptic/critic/`, truth and analysis, Testbed B
  science specification, unblinding, and findings.
- **Research engineer:** `skeptic/api.py`, `skeptic/streams/`,
  `skeptic/memory.py`, `proposals/`, and Experiment 2 (`skeptic/analysis/exp2.py`).
- **Software engineer:** `modal_jobs/`, `skeptic/sandbox.py`,
  `skeptic/store.py`, `skeptic/ledger.py`, `skeptic/rules/`, `dashboard/`,
  and the Makefile/reproduction workflow.

The proposal API, ledger/decision rows, and verdict function are the shared
interfaces described on slides 8-11. These files are placeholders, so their
presence does not mean those contracts have been implemented or validated.

## Planned commands

Run `make help` for the full command list from slide 22 and the associated work
packages. `setup`, `train40`, `sanity`, `smoke`, `stream`, `exp1`, `exp2`,
`seal-hash`, `unblind`, `report`, `reproduce`, and `testbed-b` are reserved targets.
No command currently launches compute, writes decisions, or unseals data.

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

`PREREGISTRATION.md` is a draft template, not a frozen registration. Ryu must
confirm the proposed defaults and record the preregistration commit before a
main run. Keep sealed test arrays and scores outside git, the public ledger,
shared memory, and agent inputs. E4 must implement and verify the guard and
decision-hash checks before any unblinding workflow is enabled.
