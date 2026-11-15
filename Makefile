.DEFAULT_GOAL := help

UV ?= uv
PIPELINE_TARGETS := smoke stream exp2 unblind report reproduce testbed-b

.PHONY: help bootstrap setup setup-check test train40 sanity e3-plan e3-canaries seal-hash seal-check exp1 exp1-plan $(PIPELINE_TARGETS)

help:
	@printf '%s\n' \
	  'Skeptic: E1-E3 validation and E4 store/freeze checks are implemented.' \
	  '' \
	  'bootstrap  Install locked Python dependencies and initialize pinned ER-1' \
	  'setup      Provision Modal app and two volumes; validate named secrets and mounts (E1)' \
	  'setup-check Verify an existing deployment without creating resources' \
	  'test       Run local project tests (no Modal credentials needed)' \
	  'train40    Launch forty E2 fine-tunes on Modal B300s' \
	  'sanity     Audit 40 open caches and report validation AUROC (E2)' \
	  'e3-plan    Preview four canary static checks locally (E3)' \
	  'e3-canaries Run the four canaries on one CE validation cell (E3 CPU only)' \
	  'smoke      Full canary verdicts and sealed guard (R1, L2, E3, E4; pending)' \
	  'stream     Templated and LLM proposal streams (L3, L5)' \
	  'exp1       Replay registered settings supplied via REPLAY_ARGS (E5)' \
	  'exp1-plan  Validate replay configuration without writing decisions' \
	  'exp2       Closed loop with three arms (L6)' \
	  'seal-hash  Freeze complete E5 decision logs after preregistration (E4)' \
	  'seal-check Verify a published decision freeze before test reads (E4)' \
	  'unblind    Guard check and truth tables (R5, E4)' \
	  'report     Figures, tables, and scorecard (R5)' \
	  'reproduce  Replay the ledger to regenerate every table (E8)' \
	  'testbed-b  Probe-deferral stretch testbed (R4, E6)'

bootstrap:
	$(UV) sync --frozen
	git submodule update --init third_party/er1

setup: bootstrap
	$(UV) run --frozen python -m modal_jobs.setup

setup-check:
	$(UV) run --frozen python -m modal_jobs.setup --check-only

test:
	$(UV) run --frozen python -m pytest tests -q

train40:
	$(UV) run --frozen python -m modal_jobs.train40 run --all

sanity:
	$(UV) run --frozen python -m modal_jobs.sanity

e3-plan:
	$(UV) run --frozen python -m modal_jobs.sandbox plan --canaries

e3-canaries:
	$(UV) run --frozen python -m modal_jobs.sandbox run --canaries

exp1-plan:
	$(UV) run --frozen python -m skeptic.rules.replay plan $(REPLAY_ARGS)

exp1:
	$(UV) run --frozen python -m skeptic.rules.replay run $(REPLAY_ARGS)

seal-hash:
	$(UV) run --frozen python -m skeptic.store freeze

seal-check:
	$(UV) run --frozen python -m skeptic.store unseal-check

# Replace each target with its implementation in the corresponding work package.
# A placeholder must fail rather than report a successful experiment or unblind.
$(PIPELINE_TARGETS):
	@printf '%s\n' 'Not implemented: make $@. See make help and the design-doc work packages.' >&2
	@exit 2
