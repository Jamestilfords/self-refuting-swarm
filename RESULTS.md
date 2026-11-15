# Skeptic: results and status

**Read this first.** Every number below comes from validation data. The sealed test set was never opened.

## The question

An autoresearch loop that keeps "whatever scored higher on validation" will always find something to keep.
Does it find real improvements, or noise? We let AI agents propose out-of-scope detectors and judged the
same proposals with three acceptance rules: R0 ratchet (keep if better), R1 Thresholdout, and R2, a
code-only statistical critic with six gates and online false-discovery control (LORD++).

## What we found (validation side)

Experiment 1: 487 candidate proposals (310 templated + 181 LLM-written, minus registered baselines and
canaries), 200 random orderings per cell, 4 cells.

| Cell | R0 ratchet accepts per ordering | R1 Thresholdout | R2 critic |
|---|---|---|---|
| Banking77-OOS / BERT | 6.5 | 28.9 | 0 |
| Banking77-OOS / RoBERTa | 6.6 | 33.0 | 0 |
| CLINC150 / BERT | 5.4 | 66.4 | 0 |
| CLINC150 / RoBERTa | 6.0 | 37.6 | 0 |

- **The ratchet accepts about what ordering alone produces.** A random ordering of 487 items has about
  6.8 "new best so far" moments. The ratchet accepted 5.4–6.6.
- **Most of its accepted gains are tiny.** On CLINC150/BERT, 85% of accepted steps gained less than 0.001
  AUROC and 16% were better in all 5 seeds (exploratory).
- **The critic accepted nothing in all 800 replays.** Its verdicts were mostly SCORE-MATCHED and NO EFFECT.
  The best candidate in every cell was UNDERPOWERED in all 200 orderings; on Banking77/BERT it beat the
  strongest baseline by +0.0055 (p = 0.0045) against a multiplicity bar of at most 0.0027.
- **Canaries:** four planted fakes (rescaled baseline, renamed baseline, noise, test-leak attempt) received
  their four planted verdicts in all four cells (16 of 16).

Experiment 2: a closed loop (propose → score on validation → rule → memory → next proposal),
18 unattended runs × 80 proposals = 1,440 proposals in 1 h 46 min.

- Ratchet runs kept 25 proposals in total; critic runs kept 0.
- **H5 is a null result.** Shared memory did not reduce repeated failures, and the registered "wasted
  proposal" measure saturates (0.94–1.00 in every arm), so it cannot separate the arms.

Full tables: [`tables/validation_summary.md`](tables/validation_summary.md).

## One proposal through the loop (saved run `runs/exp2/banking77_bert/B0`)

1. **Scout (Exa search)** saved source card `m_0130`: "A Simple Fix to Mahalanobis Distance for Improving
   Near-OOD Detection" (https://arxiv.org/abs/2106.09022), with the method written out.
2. **Critic** judged the first 50 proposals of the run; none passed.
3. **Analyst** read those verdicts and wrote lesson `m_0115`: fusion and kNN proposals keep reproducing the
   incumbent ranking, density variants split between score-matching and negative gains; avoid more
   monotone blends.
4. **Proposer** wrote proposal `969fe13dd999` (slot 60), citing both the source card and the lesson: a
   class-conditional fusion of the bottleneck and encoder densities.
5. **Modal** scored it on validation (5 seeds): +0.0019 AUROC over the incumbent.
6. **Critic** verdict: SCORE-MATCHED at gate G2, because it did not beat the strongest registered baseline.

Across the six memory-arm runs the Scout saved 103 source cards and the Analyst wrote 13 lessons; later
proposals cited a source card 97 times and a lesson 68 times. This shows the agents used each other's
output. It does not show that doing so helped: H5 is a null.

## What we did not do

- **The test set was not opened.** FDP, recall, the winner's-curse gap and hypotheses H1–H4 are **not
  computed**. We therefore do not claim that the ratchet's accepted proposals are false discoveries. It is
  also possible that the critic is too strict; only the sealed test can settle that.
- Test scoring of the accepted proposals through the sandboxed sealer did not fit the remaining time, and we
  chose not to do a partial or unguarded read.
- Testbed B was not run.
- **Zero critic acceptances is not evidence of fewer false discoveries.** Under the pre-registration, the
  paired H2 comparison excludes orderings where a rule accepts nothing, so no comparison is eligible.
- **LORD++ is very strict here.** With no discoveries its level is 0.0027 for the first test, 0.0006 for
  the second and below 0.0002 by the tenth, across 100–400 tests per ordering. One candidate on
  CLINC150/BERT had p = 0.0006 and was never accepted. Whether a real improvement can pass at this stream
  length was not tested on real data (the unit tests accept a planted large improvement).

## Disclosures

- The pre-registration was committed late (commit `5b0e6b2`, 2026-11-15 09:52 UTC), after validation
  scores existed. It says so and lists what had been seen.
- Validation scores came from a batch scorer with a static validator, not from the isolated sandbox. The
  sandboxed scorer was smoke-tested only.
- Both Modal volumes, including the sealed one, are in one team member's workspace. Nobody read the sealed
  test arrays; the only sealed read was validation labels for the calibration gate.
- A 9-proposal Experiment 2 smoke run (`runs/exp2_smoke`) happened before registration and is excluded.
- 108 further LLM proposals were written after the stream was frozen; they are in
  `runs/ledger_608_exploratory.jsonl` and are not part of Experiment 1.
- Gate G6 compares a fusion with the best linear mix of its parents and can flag an honest non-linear fusion.
- R1 accepting more than R0 is reported as observed; we have not ruled out a settings artefact.

## Reproduce the tables

The replay decision logs (600 files) are in git history at commit `2f96c70`
(decision digest `634e3e0ca73a8134f0d806de575a5df5b22cdd9764986ff0bc93b6beb38472df`); they were removed
from the working tree for size.

```sh
git clone https://github.com/Jamestilfords/self-refuting-swarm.git && cd self-refuting-swarm
mkdir -p /tmp/dec && git archive 2f96c70 runs/decisions | tar -x -C /tmp/dec
uv run --frozen python -m skeptic.analysis.validation_summary --decisions /tmp/dec/runs/decisions
make dashboard        # browse the same results at http://localhost:8501
```

## What is in the system

- **Four agents:** Scout (Exa web search, source cards with URLs), Proposer (writes score functions),
  critic (code only, no LLM), Analyst (writes a lesson every 25 proposals). They share one memory, mirrored
  to past.dev (313 entries ingested and 310 recalls in the Experiment 1 stream).
- **Modal:** 40 fine-tunes on B300 GPUs, validation scoring of 608 proposals × 4 cells × 5 seeds, and the
  1,440-proposal closed loop.
- **Guard:** test arrays sit in a sealed volume that code may read only after the decision logs are hashed
  and published.

## To finish the registered analysis

1. Publish the decision digest on main (`make seal-hash`).
2. Test-score the accepted proposals with the sealer (`python -m modal_jobs.seal run …`).
3. Run the truth and hypothesis code in `skeptic/analysis/` once, and report H1–H4.
