.DEFAULT_GOAL := help

UV ?= uv
PIPELINE_TARGETS := smoke exp2 testbed-b
SEALED_ROOT ?= sealed
WORKERS ?= 0

.PHONY: help bootstrap setup setup-check test train40 sanity e3-plan e3-canaries seal-hash seal-check exp1 exp1-plan unblind fetch-sealed report reproduce agents-bootstrap agents-plan agents-check agents-test stream dashboard dashboard-test $(PIPELINE_TARGETS)

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
	  'agents-bootstrap Install locked agent SDK dependencies and pinned ER-1' \
	  'agents-plan Preview the Scout/Proposer workflow without API calls' \
	  'agents-check Check local agent configuration (does not authenticate)' \
	  'agents-test Run offline tests for the agent workflow' \
	  'stream     Run bounded Exa/Proposer workflow in Modal CPU sandboxes (L5)' \
	  'exp1       Replay registered settings supplied via REPLAY_ARGS (E5)' \
	  'exp1-plan  Validate replay configuration without writing decisions' \
	  'exp2       Closed loop with three arms (L6)' \
	  'dashboard  Serve the Streamlit validation demo at localhost:8501 (E7)' \
	  'dashboard-test Test the dashboard data adapter and Streamlit controls' \
	  'seal-hash  Freeze complete E5 decision logs after preregistration (E4)' \
	  'seal-check Verify a published decision freeze before test reads (E4)' \
	  'unblind    Hash the sealed volume and write runs/unblind.json (E6)' \
	  'fetch-sealed  Download sealed scores into ./sealed after unblind' \
	  'report     Write Experiment 1 tables from runs/unblind.json (E6)' \
	  'reproduce  Same as report; no training and no new scoring' \
	  'WORKERS=N  Parallel truth workers for report and reproduce (0 uses every core)' \
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

agents-bootstrap:
	$(UV) sync --frozen --group agents
	git submodule update --init third_party/er1

agents-plan:
	$(UV) run --frozen --group agents python -m skeptic.agents plan $(AGENT_ARGS)

agents-check:
	$(UV) run --frozen --group agents python -m skeptic.agents check $(AGENT_ARGS)

agents-test:
	$(UV) run --frozen --group agents python -m pytest tests/test_agents.py tests/test_agent_runtime.py -q

stream:
	$(UV) run --frozen --group agents python -m skeptic.agents run $(AGENT_ARGS)

dashboard:
	$(UV) run --frozen --only-group dashboard streamlit run dashboard/app.py $(DASHBOARD_ARGS)

dashboard-test:
	$(UV) run --frozen --only-group dashboard --only-group dev python -m pytest tests/test_dashboard.py tests/test_dashboard_evidence.py tests/test_dashboard_app.py -q

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

unblind:
	$(UV) run --frozen python -m skeptic.analysis.unblind

fetch-sealed:
	@test -f runs/unblind.json || { echo "run make unblind first" >&2; exit 1; }
	mkdir -p $(SEALED_ROOT)
	modal volume get --env $${MODAL_ENVIRONMENT:-main} skeptic-sealed scores $(SEALED_ROOT)/
	modal volume get --env $${MODAL_ENVIRONMENT:-main} skeptic-sealed plain_ece.json $(SEALED_ROOT)/

report reproduce:
	PYTHONPATH=third_party/er1/src $(UV) run --frozen python -m skeptic.analysis.report --sealed $(SEALED_ROOT) --workers $(WORKERS)

# Replace each target with its implementation in the corresponding work package.
# A placeholder must fail rather than report a successful experiment or unblind.
$(PIPELINE_TARGETS):
	@printf '%s\n' 'Not implemented: make $@. See make help and the design-doc work packages.' >&2
	@exit 2
