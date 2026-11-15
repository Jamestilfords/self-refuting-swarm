# Guarded Modal test scoring

The trusted sealer verifies the published main commit, downloads the frozen
preregistration, manifest and decision logs from that exact commit, and
recomputes the decision digest. It also checks that the proposal's content
hash is in that cell's frozen proposal set. Only then can it read test arrays.
Network failures, changed main, draft registration, corrupt logs or mismatched
hashes fail closed. Verification uses the public GitHub API/raw file endpoints.
Downloads are bounded (16 MiB per manifest/log, 1 GiB of logs total).

Proposal code executes in fresh network-blocked gVisor sandboxes without
volumes or secrets. Each of five seeds is checked against a hidden subset.
Inputs contain training labels and anonymously shuffled test features; test
labels and ID/OOS membership are not sent to the sandbox. Worker stdout/stderr
are discarded. The trusted wrapper restores item order and saves scores with
the proposal hash, decision commit and digest in `/sealed/scores/<group>/<sha>.npz`.
The public response contains only `sha`, `cell`, and `status`.

## Local checks (no Modal jobs)

```sh
make bootstrap
uv run --frozen python -m pytest tests/test_seal.py tests/test_store.py tests/test_e3.py -q
uv run --frozen python -m modal_jobs.seal --help
```

Expected: tests pass with synthetic caches and mocked cloud/network calls.
They check blocked pre-freeze reads, complete remote-proof verification,
proposal membership, five aligned seed outputs, no test labels in the sandbox
input, atomic publication, safe resume and status-only responses.
A successful mocked call returns `sealed`; a second identical call returns
`already_sealed` without recomputing scores. A changed provenance is rejected.

## Preview a selected proposal

Replace `<sha>` with a proposal filename present in the checkout:

```sh
uv run --frozen python -m modal_jobs.seal plan \
  --proposal proposals/<sha>.py --dataset clinc150 --backbone bert
```

Expected: JSON naming the proposal, cell and arm. No freeze proof, test read,
network access or cloud job happens during `plan`.

## Run only after the experiment freeze is published

Prerequisites: completed preregistration, reviewed complete decision logs and
manifest, and `runs/DECISIONS.sha256` committed and published on main. Use a
clean checkout at that published main commit and the Modal profile containing
the sealed caches. The proposal must be in the frozen decision set.

```sh
make seal-check
uv run --frozen python -m modal_jobs.seal run \
  --proposal proposals/<sha>.py --dataset clinc150 --backbone bert
```

Expected: `seal-check` prints the published commit and decision digest. The
Modal call returns `sealed` or `already_sealed`; it prints no test AUROC or
per-item scores. Scores remain in the sealed volume for the separately approved
unblinding analysis. `run` launches CPU work and uses Modal credits; it does
not retrain a model.

These PR checks do not perform a real freeze, launch Modal jobs, or validate
cloud execution. The unblinding/report command remains separate research work.
