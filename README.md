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

## Experiment 1 truth and metrics

`skeptic/analysis/` is the pre-registered answer key for Experiment 1. It reads decision logs and, after unsealing, scores accepted changes on test data. It does not re-run the ratchet, Thresholdout, or the critic. `python -m skeptic.analysis.config --markdown` prints the definitions for `PREREGISTRATION.md` §4. Run it with `PYTHONPATH=third_party/er1/src` when you import the scoring code; the config table itself does not need ER-1.

`paired_bootstrap_fast` is an equivalence-tested vectorisation of ER-1's `paired_bootstrap` at commit `e877e9159d`. It draws the same resample indices and returns the same tie-aware weighted AUROC. Truth intervals use B = 5,000. The first truth pair in a process is checked against ER-1 at B = 20.

Defaults still to be confirmed in the preregistration:

- **D1.** Recall's numerator is the accepted proposals that sit in S_cell, the set of real improvements over CE energy. The numerator and the denominator use that same set, so recall cannot exceed 1.
- **D2.** Intervals use `paired_bootstrap_fast`, bit-identical to ER-1, with B = 5,000.
- **D3.** The two-level bootstrap reweights accepted changes. It does not replay the rules.
- **D4.** H2's p-value is the third-smallest per-cell p-value. H3's p-value is the bootstrap probability that the recall ratio is at most 0.5.
- **D5.** Final-pick ECE uses the proposal's in-scope test probabilities when those were stored, otherwise the plain-model ECE of its arm. The best single density is chosen by validation AUROC.
- **D6.** Strict truth skips a cell whose scores are missing and does not call that a reversal. A reversal is a combined 95% interval lying entirely below 0.
- **D7.** Truth is the gain over the incumbent at acceptance, even when the critic compared the proposal with a baseline rather than that incumbent.
- **D8.** An ordering where either the ratchet or the critic accepted nothing is left out of H2's paired difference, and the count of those orderings is reported.

A finite test set misses some real gains. In the low-power synthetic check (`delta_auc = 0.005`) no planted improvement cleared a 95% interval above zero, so S_cell was empty and measured recall was undefined rather than written as 0. When that recall is defined, the test requires it to be at most the recall against the planted effects.

## Research state

`PREREGISTRATION.md` is a draft template, not a frozen registration. @mid9t must
confirm the proposed defaults and record the preregistration commit before a
main run. Keep sealed test arrays and scores outside git, the public ledger,
shared memory, and agent inputs. E4 must implement and verify the guard and
decision-hash checks before any unblinding workflow is enabled.
