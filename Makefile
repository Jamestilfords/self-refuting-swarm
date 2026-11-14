.DEFAULT_GOAL := help

PIPELINE_TARGETS := setup train40 sanity smoke stream exp1 exp2 seal-hash unblind report reproduce testbed-b

.PHONY: help $(PIPELINE_TARGETS)

help:
	@printf '%s\n' \
	  'Skeptic: repository scaffold. Pipeline targets are not implemented yet.' \
	  '' \
	  'setup      Modal app, secrets, and two volumes (E1)' \
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

# Replace each target with its implementation in the corresponding work package.
# A placeholder must fail rather than report a successful experiment or unblind.
$(PIPELINE_TARGETS):
	@printf '%s\n' 'Not implemented: make $@. See make help and the design-doc work packages.' >&2
	@exit 2
