# Guarded Modal test scoring

Standalone trusted sealer calls verify the published freeze, download the frozen
preregistration, manifest and decision logs from that exact commit, and
recomputes the decision digest. It also checks that the proposal's content
hash is in that cell's frozen proposal set. Only then can it read test arrays.
Network failures, changed frozen inputs, draft registration, corrupt logs or mismatched
hashes fail closed. Verification uses the public GitHub API/raw file endpoints.
Downloads are bounded (16 MiB per manifest/log, 8 GiB of logs total to accommodate all eight registered rule variants across 200 orderings).

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
uv run --frozen --group dev python -m pytest tests/test_seal.py tests/test_seal_batch.py tests/test_store.py tests/test_e3.py tests/test_score_stream.py -q
uv run --frozen python -m modal_jobs.seal --help
```

Expected: tests pass with synthetic caches and mocked cloud/network calls.
They check blocked pre-freeze reads, complete remote-proof verification,
proposal membership, five aligned seed outputs, no test labels in the sandbox
input, atomic publication, safe resume and status-only responses.
Both CLINC150 and Banking77, with BERT and RoBERTa, are covered. Banking77's
training/recipe identifier is `banking77_oos`; its storage group remains
`banking77_<backbone>`. The CLI also accepts `--dataset banking77` as an alias.
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

## Preview a batch (offline)

```sh
uv run --frozen python -m modal_jobs.seal plan --all
uv run --frozen python -m modal_jobs.seal plan --all \
  --dataset banking77 --backbone bert --batch-size 10
```

`--all` selects each unique proposal/cell from `runs/decision_manifest.json`.
It does not select proposals from later exploratory ledgers. Optional dataset
and backbone filters narrow the selected cells. Source files are checked for
their content hash and static metadata without executing them. Missing or
changed sources abort the entire plan before Modal starts.

The JSON preview lists identities, job count, chunk count, and the maximum
number of fresh sandboxes (ten per uncached job: full/subset for five seeds).
`freeze_verified: false` is intentional: previewing is not authorization to
read test data. A missing manifest stops `--all`; restore/prepare the reviewed
freeze artifacts through the experiment's publication workflow first. A
single-proposal preview does not require the manifest.

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

## Batch scoring after publication

```sh
uv run --frozen python -m modal_jobs.seal run --all
```

The same filters and `--batch-size` work with `run`. The default and maximum
chunk size is 25 proposal/cell jobs, with at most 40 batch containers running
concurrently. The trusted local coordinator verifies the entire clean published
checkout once and creates an ephemeral, workspace-authenticated Modal Dict
containing the freeze identity and authorized proposal/cell pairs. Each batch
checks this receipt before reading test arrays; batches perform no GitHub downloads.
The Dict is kept alive for the run and is never given to proposal sandboxes.
Each container processes its chunk and seeds sequentially. Every proposal and seed still gets fresh isolated
workers; batching does not share a proposal process. Each sandbox allows five
minutes and its worker process allows 270 seconds. The registered 60-second
`score(ctx)` validity limit is unchanged. A five-seed scoring budget is 55 minutes;
the single-proposal sealer adds a 30-minute freeze-verification allowance (85
minutes total). A full 25-job batch has a 23-hour 25-minute timeout, below Modal's
24-hour function limit. These are worst-case budgets, not expected runtimes.
Standalone calls retain bounded GitHub proof downloads. Batch runs use the
already-verified local checkout and one `git ls-remote` check, avoiding 79 full
GitHub downloads (over 300 GiB). Keep the coordinator terminal open.

The registered digest is pinned to freeze commit `0a48160cadf6127b47569607c87f49a806aab11e`.
Runner-only updates retain that decision identity only if the published decision
directory, manifest, hash and preregistration Git objects are identical to the
anchor. Thus the existing smoke result returns `already_sealed`, without deletion,
restamping or recomputation. Any changed digest, proposal hash or unrelated commit
still fails the provenance check. The analysis/unblinding guard uses the same identity.

At 40 batches with sequential seeds, up to 40 sandbox workers accompany 40
trusted containers. Workspace quotas and other running apps can reduce actual
concurrency. James's single-job timing suggests roughly 10 hours of scoring
at this concurrency, before startup and stragglers; benchmark the first wave
before promising a seven-hour completion. Five-seed parallelism is deliberately
disabled because it would multiply sandbox concurrency and memory demand.

Progress is printed after each chunk is collected, in submission order. A slow
earlier chunk may delay progress messages while later chunks finish. Statuses
are appended to `runs/seal/status.jsonl` (ignored by Git), including the freeze
commit and digest. `--report PATH` selects another destination; keep it outside
tracked files so it cannot dirty the freeze checkout. No test metrics, arrays,
worker output, or raw exception messages are printed or written there.

- `sealed`: all five seed outputs were checked and the file was committed to
  the sealed volume.
- `already_sealed`: an existing artifact passed schema and matching
  proposal/commit/digest checks; scoring was skipped.
- `failed`: the proposal failed, the batch failed, or its response was incomplete.
  Exit status is 1 if any job failed; argument errors use 2; success uses 0.

Rerun the same command at the same published freeze to resume. The remote sealed
artifacts, not the local status report, determine what can be reused. A lost or
interrupted client can leave completed remote artifacts without a local status
line; these are checked again on resume. A changed provenance is rejected, never
overwritten. Use one coordinator for a scoring run. This CLI must remain running;
it is not a detached scheduler. Use a persistent terminal for an unattended run.

## Handoff to the truth/report implementation

Batch scoring alone does not finish H1–H4. Before launching the real run, resolve:

1. **Freeze publication:** complete decision logs, manifest and digest must be
   available and published on main. This change does not recreate, modify, or
   publish those artifacts.
2. **Reference authorization:** the current guard authorizes proposal hashes in
   the manifest's `proposals` lists only. The registered candidate stream excludes
   baseline hashes. CE-energy is needed as the recall reference and starting
   incumbent, so the team must establish its authorized scoring path. This change
   does not silently add baselines or expand the sealed-read allowlist.
3. **Full recall coverage:** accepted proposals alone are insufficient to compute
   the registered stream truth set. `--all` covers the manifest candidates, but
   still needs the reference scores above.
4. **Analysis format:** sealed outputs are `scores/<group>/<sha>.npz` with a
   `(5, n_items)` array and `n_in`. Ryu's folder source currently expects separate
   `seed*.npy` arrays under `<cell>/<name>/`; it needs a guarded adapter/export and
   baseline-name mapping. Probability/calibration outputs needed for the wider
   report are also separate from this score-only format.

Successful batching is not an unblinding/report command. Test scores remain
sealed until the separately reviewed analysis is run.

These PR checks do not perform a real freeze, launch Modal jobs, or validate
cloud execution. The unblinding/report command remains separate research work.
