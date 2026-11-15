# Sealed store and decision freeze

`skeptic.store` provides a read guard for trusted jobs and commands to freeze
and verify decision logs. It depends only on the existing NumPy dependency
and Python's standard library. These commands do not start Modal jobs, score
proposals, or create a Git commit.

This change publishes the guard API and freeze tooling. The existing legacy
scoring functions still need their separate isolation integration; they do
not automatically become protected by importing this module. Proposal code
must run in a separate Sandbox without a sealed-volume mount. A Python read
guard cannot restrict code that already has direct filesystem access.

## Required freeze inputs

`PREREGISTRATION.md` must contain a line beginning with `Status: FROZEN`
(optional Markdown bold markers are accepted) and contain no `TODO:` entries.
The registration must be completed and approved before this status is set.

The decision producer must supply:

- `runs/decisions/R0/ord_000.jsonl`, and the same layout for R1 and R2 and every
  registered ordering.
- `runs/decision_manifest.json`, with `schema_version: 1`, `rules:
  ["R0", "R1", "R2"]`, a positive integer `orderings`, a sorted `cells` list,
  the complete `files` list, and a `proposals` map from each cell to its sorted,
  unique proposal hashes.

The four cells are `banking77_oos/bert`, `banking77_oos/roberta`,
`clinc150/bert`, and `clinc150/roberta`. Every decision file must cover all four.
Rows need `schema_version`, `rule`, `cell`, `ordering`, contiguous zero-based
`idx`, the 12-character proposal `sha`, `incumbent`, and `verdict`.

The guard checks matching proposal sequences across rules, a shared starting
incumbent per cell, and an incumbent chain that advances only on `ACCEPT`.
R2 accepts require the critic's G1, G2, G3, G5 and G6 outcomes to be `pass`;
G4 is the critic's classification of a failed G3 test. Missing files, symlinks,
duplicate proposals, and mismatched manifests are rejected.

The operational `runs/decisions/e3.jsonl` file is outside this freeze set.
The E5 replay engine that produces these files is a separate change.

## Commands

After the registration and complete decision logs are ready:

```sh
uv run --frozen python -m skeptic.store digest
make seal-hash
uv run --frozen python -m skeptic.store verify
```

`seal-hash` creates `runs/DECISIONS.sha256` once and refuses to overwrite it.
It contains a single lowercase SHA-256 digest followed by a newline. The
digest starts with the domain bytes `skeptic-decisions-v1\0`; for each sorted
decision path it hashes the UTF-8 relative path and exact file bytes, each
preceded by its unsigned 8-byte big-endian length. It finishes with
`manifest\0`, the 8-byte manifest length, and its exact bytes.

Review the freeze artifacts and publish them with the completed
preregistration on the registered repository's `main`. Then:

```sh
make seal-check
```

`seal-check` requires a clean checkout whose HEAD equals GitHub `main`.
It checks the registered origin URL, recomputes the decision hash, and compares
the exact bytes of the hash, manifest, every decision log, and preregistration
against that commit. Ignored or untracked logs cannot substitute for published
files. A failed network or Git check leaves test access locked.

The publication check reads Git metadata and decision artifacts only. The
`unblind` analysis target remains a placeholder.

## Trusted-job API

```python
from skeptic.store import SealedStore

store = SealedStore("/sealed", "/path/to/frozen/checkout")
labels = store.read_array("clinc150_bert/ce_seed0", "val", "labels")

store.unseal()  # Raises GuardError unless the published freeze verifies.
test_logits = store.read_array("clinc150_bert/ce_seed0", "test", "logits")
```

Training and validation arrays can be read before the freeze. `test` and
`oos_test` require a verified permit. Run names and fields are allowlisted,
cache symlinks are rejected, and NumPy pickle loading is disabled.
A failed attempt to reverify the freeze revokes any previous permit.
The permit is scoped to that store instance; it is a workflow check for
trusted code, not a restriction on a Modal workspace administrator.
