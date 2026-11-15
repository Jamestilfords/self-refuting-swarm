.DEFAULT_GOAL := help

UV ?= uv
PIPELINE_TARGETS := train40 sanity smoke stream exp1 exp2 seal-hash unblind report reproduce testbed-b

.PHONY: help bootstrap setup setup-check test $(PIPELINE_TARGETS)

help:
	@printf '%s\n' \
	  'Skeptic: E1 setup is implemented; experiment pipeline targets are pending.' \
	  '' \
	  'bootstrap  Install locked Python dependencies and initialize pinned ER-1' \
	  'setup      Provision Modal app and two volumes; validate named secrets and mounts (E1)' \
	  'setup-check Verify an existing deployment without creating resources' \
	  'test       Run local infrastructure tests (no Modal credentials needed)' \
	  'train40    Forty fine-tunes and caches (E2)' \
	  'sanity     Baseline AUROC against the paper (E2)' \
	  'smoke      Four canaries, end to end (R1, L2, E3, E4)' \
	  'stream     Templated and LLM proposal streams (L3, L5)' \
	  'exp1       Three acceptance rules across matched orderings (E5)' \
	  'exp2       Closed loop with three arms (L6)' \
	  'seal-hash  Freeze and hash decision logs (E4)' \
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
	$(UV) run --frozen python -m unittest discover -s tests -v

# Replace each target with its implementation in the corresponding work package.
# A placeholder must fail rather than report a successful experiment or unblind.
$(PIPELINE_TARGETS):
	@printf '%s\n' 'Not implemented: make $@. See make help and the design-doc work packages.' >&2
	@exit 2
