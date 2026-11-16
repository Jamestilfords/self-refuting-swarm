# Handoff: from here to the unblinded results

Status as of Sun 15 Nov 23:10 PT. Steps follow Ryu's unblinding plan (PR #12). Modal steps run in James's
workspace `jamestilfordsusanto`, which holds both volumes (`skeptic-open`, `skeptic-sealed`); Ryu also has
owner access.

| # | Step | Owner | State |
|---|---|---|---|
| 1 | Merge PR #12 | Ryu | Done 20:56 |
| 2 | `check_references`: open validation scores = ER-1 baseline arrays | James | **Passes** with the float32 fix below |
| 3 | Re-run the replay with the PR code, commit `runs/decisions/` and the manifest with `references`, `make seal-hash`, push | Xay | **Open.** Blocks everything after it |
| 4 | `make seal-check`, `python -m modal_jobs.seal plan --all --batch-size 25`, then `run --all` | James (Modal) | Waits on 3. Expect 1,968 jobs in 79 batches, several hours; terminal must stay open |
| 5 | Implement the test-side plain ECE job | open | `python -m modal_jobs.plain_ece` writes `/sealed/plain_ece.json` on the volume |
| 6 | `python -m modal_jobs.plain_ece`, `make unblind`, commit `runs/unblind.json` and `runs/guard.log`, `make fetch-sealed`, `make report` | James (Modal) with Ryu | Waits on 4 and 5. Unblind hashes the volume; report reads the download in `./sealed` |

## Step 2 detail

Reference arrays: `python -m modal_jobs.reference_baselines` writes the 9 registered baselines per cell to
`/open/baselines/<group>/<name>/seed<k>.npy`, reading only `train`, `val`, `oos_val` from the sealed caches
(validation labels are needed for plain ECE). Download with
`modal volume get skeptic-open baselines runs/ --force`, then run
`python -m skeptic.analysis.check_references`.

The first run failed on `clinc150/bert energy_ce`. Diagnosis (all 20 reference/cell pairs): every stored
array equals the ER-1 array rounded to float32; minimum Spearman 0.9999999995; up to 11 float32 ties per
pair. The score files are stored as float32, and rounding turns near-equal values into ties that flip rank
order under an exact 1e-12 rank test. The check now also passes a seed when the stored array equals the
reference rounded to float32 (`tests/test_check_references_float32.py`); a real reordering still fails.

## Where things are

| What | Where |
|---|---|
| Registered validation ledger (500 proposals, 2,000 rows) | `runs/ledger.jsonl` (sha256 `a484470a…`) |
| Per-item validation scores | open volume `scores/<group>/<sha>.npz` (float32, shape 5 × n, first `n_in` in-scope); local copy `runs/scores/`, hashes in `runs/score_manifest.sha256` |
| 108 extra LLM proposals, not in Experiment 1 | `runs/ledger_608_exploratory.jsonl` |
| Experiment 1 decisions published once | git commit `2f96c70` (digest `634e3e0c…`), reverted at 12:19 |
| Validation-side results tables | `tables/validation_summary.md`, `python -m skeptic.analysis.validation_summary` |
| Experiment 2 (18 closed-loop runs) | `runs/exp2/`, `python -m modal_jobs.exp2 report` |
| One traced proposal, sponsor evidence | `tables/evidence_bundle.json`, `python -m skeptic.analysis.evidence_bundle export` |
| Results and disclosures | `RESULTS.md` |

## Groups and names

Cells are `clinc150/bert`, `clinc150/roberta`, `banking77_oos/bert`, `banking77_oos/roberta`. Volume
folders drop the `_oos`: `clinc150_bert`, `clinc150_roberta`, `banking77_bert`, `banking77_roberta`.
Training recipes use the dataset id `banking77_oos`. Run folders are `<arm>_seed<k>`, arms `ce` and `edl`,
seeds 0–4. Score file names are the first 12 hex digits of the proposal file's SHA-256.
