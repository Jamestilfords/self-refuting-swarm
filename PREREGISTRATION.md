# Skeptic preregistration

**Status: Registered. Commit 5b0e6b223c60a3bab3ddfd058e12caa818330a7c, 2026-11-15 09:52:44 UTC.**

Owner: [@mid9t](https://github.com/mid9t) (Ryu, R3).
Sources: the Skeptic design document (slides 5–21 and 31, with speaker notes), the R1 and R2 implementation specs, and the ER-1 paper draft *Where Does the Evidence Come From? Auditing Density-Aware Evidential Learning for Out-of-Scope Intent Detection* (Oct 20, 2026), with ER-1 at commit `e877e9159d`.

Once this file is committed, no hypothesis, threshold, gate, rule, truth definition or analysis in it changes. Anything added or changed afterwards is labelled **exploratory** in the README and in the pitch, and the original analysis is still reported. Record the commit hash of this file in the README.

---

## 1. Hypotheses and support criteria

**Question.** When an autoresearch loop searches for improvements to an OOD monitor, what share of the improvements it accepts are real? And does an audit critic built from ER-1's protocol lower that share without losing the real ones?

**Cells.** Four (benchmark × backbone): CLINC150/BERT, CLINC150/RoBERTa, Banking77-OOS/BERT, Banking77-OOS/RoBERTa.
**Rules.** R0 = ratchet, R1 = Thresholdout, R2 = critic (§3).

| ID | Status | Claim | Estimand | Supported if |
|---|---|---|---|---|
| H1 | confirmatory | Winner's curse: changes the ratchet accepts gain less on test than on validation | Mean of (Δval − Δtest) over every change R0 accepts in every ordering, both CLINC150 cells pooled; both Δ against the incumbent at acceptance | 95% CI > 0 **and** Holm-adjusted p ≤ 0.05 |
| H2 | **primary** | On the same proposals, the critic keeps fewer false discoveries than the ratchet | Per cell: mean over orderings of FDP(R0) − FDP(R2), paired within ordering | 95% CI > 0 in at least 3 of the 4 cells **and** Holm-adjusted p ≤ 0.05 |
| H3 | confirmatory | The critic keeps most of the real improvements | Mean recall(R2) ÷ mean recall(R0) over (cell, ordering) in both Banking77-OOS cells | Point estimate ≥ 0.5. The CI and Holm-adjusted p are reported either way |
| H4 | confirmatory | Validation size drives false discoveries | Mean FDP(R0) over the CLINC150 cells (100 OOS validation items) − mean FDP(R0) over the Banking77-OOS cells (730) | 95% CI > 0 **and** Holm-adjusted p ≤ 0.05 |
| H5 | secondary, descriptive | Memory reduces wasted proposals (Experiment 2) | Per run: wasted share among proposals 41–80 − wasted share among 1–40; arm B (memory) vs arm C (no memory) | Lower in arm B. Every run's value is reported; no interval is claimed |

**p-values and multiplicity. Confirmed.** All four p-values are **two-sided** bootstrap p-values, `p = max(2·min(P*(stat ≤ c), P*(stat ≥ c)), 1/B)`, with c = 0 for H1, H2 and H4 and c = 0.5 for H3. This matches the two-sided 95% intervals, as in ER-1. The hypothesis code uses the same rule: H2's single p is the third-smallest of its four per-cell p-values, and Holm is applied across H1–H4.
- H2 enters Holm as one claim. Its p is the **third-smallest** of its four per-cell p-values, so the "3 of 4 cells" rule holds at level α exactly when at least three cells have p ≤ α.
- Holm is applied across H1–H4 at 0.05.
- H5 is outside the Holm family.

**Pre-declared interpretations.**
- A null on H2 is a result: it would mean ER-1's sign-agreement numbers overstate how badly a naive loop is misled.
- If the critic accepts almost nothing on CLINC150, that is what H4 predicts. It is reported as "this benchmark cannot decide", not as a failure of the critic.
- An ordering in which either rule accepts nothing has undefined FDP. It is excluded from the paired H2 difference, and the number excluded is reported next to the result.

---

## 2. Testbeds, data splits, and exclusions

### 2.1 Testbed A (primary): 40 fine-tunes

**Grid.** 2 benchmarks × 2 backbones (`bert-base-uncased`, `roberta-base`) × 2 arms (CE, EDL) × 5 seeds (0–4) = 40 runs. Each run uses ER-1's `scripts/run_experiment.py` and `configs/default.yaml` unchanged, at commit `e877e9159d`; ER-1 enters as a git submodule pinned to that commit.

**Data** (identical to the ER-1 paper, §5.1):

| | In-scope train / val / test | OOS val | OOS test | Notes |
|---|---|---|---|---|
| CLINC150 "plus" | 15,000 / 3,000 / 4,500 (150 intents) | 100 | 1,000 | The 250 OOS training utterances are not used |
| Banking77-OOS | 5,905 / 1,506 / 2,000 (50 intents) | 730 = 530 ID-OOS + 200 OOD-OOS | 2,080 = 1,080 ID-OOS + 1,000 OOD-OOS | Files at commit `fa95533` of the authors' repository, keeping its 8 empty lines. The 2,062 ID-OOS training utterances are never loaded |

No model sees outliers during training. OOS is the positive class, and every score is oriented so that higher means more likely OOS.

**Training** (from `configs/default.yaml`):
- Inputs: 64 tokens. Batch 32, eval batch 128.
- Head: dropout 0.1, then Linear 768→256 with spectral normalization and no activation, then Linear 256→K.
- Optimizer: AdamW, learning rate 2e-5 for the encoder and 1e-3 for the head, weight decay 0.01, 10% warmup, gradient clip 1.0.
- Schedule: 5 epochs, keeping the epoch with the best validation in-scope accuracy.
- EDL arm: exponential evidence, logit clamp ±15, α offset 1e-6, Bayes-risk digamma loss, KL weight annealed to 1e-3 over 5 epochs.

**Cached outputs per run:**
- `cache.npz`: `{split}__logits`, `{split}__feats` (256-d bottleneck), `{split}__labels`, and for Banking77-OOS `{split}__oos_type` (1 = ID-OOS, 2 = OOD-OOS).
- `pooled.npz`: `{split}__pooled` (768-d).

Splits: train, val, test, oos_val, oos_test.

**Volumes.** The full cache is written to the sealed volume. Only train, val and oos_val are copied to the open volume, and the sandbox mounts only the open volume.

**Cache provenance.** Primary: fresh CUDA runs on Modal (`make train40`). †Fallback: if the 40 caches do not exist by the sanity gate, use Ryu's existing laptop caches (Apple MPS), state this in the README, and drop the claim of an independent CUDA reproduction. Actual provenance: **Modal CUDA (B300), all four cells.** The only training path in the repo writes those caches from `modal_jobs/train40.py` on B300 GPUs, and nothing in the repo records a laptop-MPS cache for any cell. The open volume was not re-read from this machine.

**Sanity check (baseline-only test read, logged)†.** `make sanity` compares CE-energy test AUROC per cell with the paper's Table 2:

| Cell | Paper CE energy |
|---|---|
| CLINC150/BERT | 0.9794 |
| CLINC150/RoBERTa | 0.9815 |
| Banking77-OOS/BERT | 0.9270 |
| Banking77-OOS/RoBERTa | 0.9294 |

**Confirmed.** Pass if every cell is within ±0.005 of these values. This read is the only test access allowed before unblinding. It computes registered baselines only and writes a line to `runs/guard.log`.

**Known limitation (declared).** The test sets of both benchmarks were analysed in the ER-1 study by the same PI. The weights here are new, but the test items and the paper's test-level conclusions are known. Those conclusions inform the scorecard expectations (§4.6) and the gate design. Test truth in this study is therefore blind to the loop and to the critic, but not to the investigator's prior knowledge of these benchmarks. Testbed B is the out-of-domain check.

### 2.2 Testbed B (stretch, exploratory pilot): probe-deferral monitor

**Status: run late, as an exploratory pilot.** The design document's go/no-go (Sat 22:00) was not met: no Llama activations or Llama Guard scores were cached by then. Under that rule Testbed B would have been dropped. It is instead run overnight Sat 14 – Sun 15 Nov, starting after this commit, and this is recorded as a deviation (§7). All Testbed B results are exploratory, whatever their outcome. If not started by Sun 04:00 PT, Testbed B is not tested and is reported as not run.

Caching activations, fitting the probe and the band, and running the exchange test use train and tune only, and may start after this commit. No held-out shift (B, C, A1) is read until `prereg/freeze_b.json` is committed and pushed. If any gated model or dataset download fails, Testbed B is dropped and that is reported.

**Monitored model and second stage:**
- `meta-llama/Meta-Llama-3-8B-Instruct`. Residual stream at the last prompt token, `hidden_states[L]` for L ∈ {8, 12, 16, 20, 24}, stored as float16.
- Probe: L2 logistic regression on standardized layer-16 activations, `class_weight="balanced"`. C is chosen from 1e-4 to 1e-1 by 5-fold CV, preferring the smallest C on ties.
- Second stage: `meta-llama/Llama-Guard-3-8B`, default S1–S14 policy. log-odds = log p(unsafe) − log p(safe) at the first generated token. Threshold fit on tune.
- Band: the window of 5% of tune items, consecutive in sorted probe logit, that minimizes weighted error. Llama Guard decides inside the band and the probe decides outside it.
- Metric: E = 5·FNR + 2·FPR(hard negatives) + 50·FPR(ordinary), with item cost = group weight ÷ group size.

**Data:**
- Train 15k: WildJailbreak vanilla plus WildChat ordinary.
- Tune 15k: WildJailbreak adversarial plus WildChat.
- †Held-out shifts: **B** = OR-Bench, 4,000; **C** = WildChat non-English and code, 4,000; **A1** = jailbreak templates × StrongREJECT requests, 2,000.
- XSTest is not touched.

**Exclusion.** `testbed_b/study_ids.txt` lists every item the five-week Activation Probes study uses, built from its data config (seed 20261026). All Testbed B items are drawn from the complement, and a test asserts zero overlap. If any gated model or dataset is unavailable, Testbed B is dropped; no substitute model is used.

**Proposals.** Signals, all oriented so that higher means defer:
- ensemble disagreement: SD of logits over 20 bootstrap probes, seeds 0–19;
- residual density: minimum squared Mahalanobis distance over 3 classes, tied covariance, Ledoit–Wolf, no PCA;
- †kNN distance to train activations, k = 10;
- †layer disagreement: SD of standardized probe logits across layers 12, 16 and 20;
- controls: token count and activation norm.

A signal is dropped if it cannot beat token count, or if its rank correlation with a kept signal is above 0.95. Two rule templates: a swap rule, and a margin-plus-signal rule with β fit on tune.

**Exchange test (critic, tune only):**
1. Net fix rate of confident items in the signal's top 1% of traffic beats that of the least useful 1% inside the band. Paired bootstrap, B = 2,000, seed 1, stratified by error group; 95% CI > 0.
2. The same holds against token count used as the signal.
3. Materiality: the rule changes at least 10% of the deferred set.
4. Ceiling: no rule beats the in-sample oracle deferral; a claim above it is a bug.
5. One-sided p-values go through LORD++ at 0.1 (§3.3).

**Freeze and decision.**
- At most two rules are written to `prereg/freeze_b.json` and committed. Only then is each held-out shift read, once, enforced by the guard.
- **Positive** if a frozen rule lowers E against the optimal 5% band, with a 95% paired-bootstrap CI excluding 0, on at least 2 of the 3 shifts.
- Directional prediction: if density helps, it helps more on C than on B.

---

## 3. Proposal streams and acceptance rules

### 3.0 Stream freeze

Proposals were generated and scored on validation **before** this file was committed (see §6). The stream is therefore frozen here by content hash, and Experiment 1 replays exactly this stream.

| Item | Value |
|---|---|
| Proposals in the ledger | 500 |
| Ledger rows (proposal × cell) | 2,000, of which 1,996 have status ok. The 4 non-ok rows are one proposal on all four cells: sha `b9cf295b6966` (the test-leak canary), status `io`, cells `banking77_oos/bert`, `banking77_oos/roberta`, `clinc150/bert`, `clinc150/roberta`. The runner detail is `attribute .load` |
| Composition | 310 templated + 181 LLM + 4 canaries + 5 baseline = 500. The other 5 are stream `baseline`, and they are not candidates: `b1e4ba2e46d5` energy@ce, `39d3e8c0e810` energy@edl, `a5325abe3321` msp@edl, `3fee8950d4d1` d_port@ce, `926c1ced9316` d_full@ce. The other four registered baselines share a sha with a templated scorecard file (`fe47292dc266` msp@ce, `6cc76a18b82b` vacuity@edl, `71faf9e6269e` d_port@edl, `2de1c4943af0` d_full@edl) and are counted in the 310 |
| Candidates per cell | N = 487. See §3.1 |
| Templated count | 310 files = the seed-0 draw of 300 plus 10 scorecard files. The draw contains none of the scorecard configurations. The eight findings are pairs, and those pairs are 10 distinct files: 7 candidates and 5 incumbents, with 2 files in both roles. 300 + 10 = 310. The plan's 308 counted 8 files for 8 findings |
| Validation scores | 5 seeds per ok row; per-item score arrays on the Modal open volume |
| Test scores | Not run at commit time (`not_run` on every row) |
| G5 applicability | 1,176 ok rows return probabilities and carry their own validation ECE; 820 do not, so G5 does not apply to them. Plain-model ECE is present on every ok row |
| `sha256(runs/ledger.jsonl)` | `a484470a07cde41908f6cf53d67b9c798eda1ae7d8ff4d5667665c6e35031a02` |
| Score-file manifest | The manifest is computed from the downloaded per-item score files and committed before the first decision file. The replay refuses to run without it. Its commit hash is recorded in §7. |

**Confirmed. No proposals are added to Experiment 1 after this commit.** The LLM stream stays at the 181 proposals in this ledger. `proposals/llm.json` lists 289 files; the 108 files that are not in the ledger are outside Experiment 1. The 181 meet the Saturday gate (≥ 100 distinct) and fall short of the planned 200–400. Results are reported for the templated and LLM streams separately and pooled, with the LLM stream's size stated.

### 3.1 Streams

**Templated** (code, no LLM). The grid of ER-1 mutations:
- feature space: bottleneck or encoder output;
- PCA: none or 128;
- Mahalanobis variant: min-class, relative, Mahalanobis++, partial;
- normalizer: minmax, qsigmoid, ecdf, fit on training log-densities;
- coupling: none, mul, add_log;
- λ ∈ {1, 2, 5, 10, 20, 40, 80, 160};
- arm.

The grid is enumerated in a fixed order, shuffled with `default_rng(0)`, and the first †**300** are taken. The templated stream is those 300 files plus 10 scorecard files (§3.0).

**LLM** (Proposer plus Scout). 200–400 proposals, with a Scout card (one Exa search) every †10 proposals. The Proposer is never shown a test number. **Distinct** means Spearman < 0.999 to every earlier proposal's validation scores in the same cell; duplicates are marked and kept out of the count. If fewer than 100 distinct LLM proposals exist by Sat 23:00, Experiment 1 runs on the templated stream plus whatever the LLM stream holds. Actual count: 181 LLM proposals in the frozen ledger (§3.0); the Sat 23:00 gate was met.

**Canaries** (excluded from every statistic and every ordering):

| Canary | Expected verdict |
|---|---|
| 2 × energy + 1 | INERT |
| CE MSP under a new name | SCORE-MATCHED |
| Gaussian noise | NO EFFECT |
| A function that reads `oos_test` | BUG (the guard raises) |

**Registered baselines are not candidates in Experiment 1.** A proposal whose score *is* a registered baseline (by registered name) is removed from the candidate list before replay, as the replay code does. The scorecard (§4.6) is exempt: its pairs use registered scores as candidates by design.

`replay_rows` retags every sha in `proposals/baselines.json` to stream `baseline`, then orderings keep only `templated` and `llm`. Vacuity `6cc76a18b82b` is in that file. It is a registered score and not a G2 baseline, and the replay drops it along with the other registered scores. Candidates per cell = status-ok proposals minus the 3 ok canaries, the 5 baseline-stream proposals, and 4 templated files that are registered scores: `fe47292dc266`, `71faf9e6269e`, `2de1c4943af0`, `6cc76a18b82b`. That is N = 487 per cell. The analysis check requires every ordering to have exactly these 487 decisions.

**Proposal validity.** A proposal is malformed if it has the wrong length, NaN, constant output, an import outside numpy/scipy/sklearn, file or network access, more than †60 s per seed, or fails the †item-wise check. The item-wise check scores a random 10% of the eval items alone and requires the same values to 1e-6. Malformed, timed-out and guard rows never enter orderings.

**PARENTS. Confirmed.** A proposal may declare `PARENTS` only when its score is a **linear fusion** of two registered scores. Couplings (mul, add_log) and other nonlinear combinations do not declare parents. Reason: G6's ceiling is a linear-fusion ceiling, and a nonlinear coupling can legitimately exceed it. The ER-1 paper shows add_log is linear in clipped energy and the *normalized* density p̂, not in the raw density score.

Of the 310 templated files, 307 declare no parents. Three scorecard couplings carry a PARENTS constant from generation: `03e0017097f5` and `cddf9272a0a9` (mul) and `2d5e3b3523bd` (add_log). The files are not edited, because that would change their hashes. Instead, the replay and the scorecard ignore PARENTS for these three shas, so G6 does not apply to them. The replay configuration lists the same three shas in `ignore_parents`.

Results are reported for the templated and LLM streams separately and pooled.

### 3.2 Orderings and starting point

†**200** orderings per cell: `numpy.random.default_rng(0).permutation`, applied 200 times over that cell's 487 candidates (§3.1). Every rule and every ablation variant replays the same permutations. The starting incumbent in every ordering is †**CE energy** (`energy_ce`). When a rule accepts, the candidate becomes the incumbent for that rule and ordering.

†**Fallback:** if the pair statistics for all orderings are not finished by Sun 05:00, the first 100 orderings are used for every rule.

### 3.3 Acceptance rules

**R0, ratchet.** Accept if the seed-mean validation AUROC of the candidate is greater than the incumbent's, by any margin.

**R1, Thresholdout** (Dwork et al., 2015, "The reusable holdout"):
- Each cell's validation set is split once, †50/50, stratified by in-scope/OOS and by utterance id (seed 0), into an open half and a guarded half.
- For a query, φ = seed-mean AUROC(candidate) − AUROC(incumbent) on each half.
- State: T̂ = T + Lap(2σ), with T = †0.01 and σ = †0.0025. Per query, draw η ~ Lap(4σ).
- If |φ_guard − φ_open| > T̂ + η, return φ_guard + Lap(σ) and redraw T̂; otherwise return φ_open.
- Accept if the returned value is > 0.
- †No budget cap. The generator is seeded by (cell, ordering index).
- **Confirmed.** The noise schedule matches Thresholdout in Dwork et al. 2015: the threshold is refreshed with Lap(2σ), the comparison uses Lap(4σ), and a held-out answer is released with Lap(σ). The paper also stops after a finite query budget; this registration sets no budget cap.

**R2, critic.** Gates in fixed order; the first failing gate gives the verdict, and a proposal that passes all of them is ACCEPT. n = number of seeds, and the **seed majority** is ⌈n/2⌉ (†3 of 5).

| Step | Gate | Fails, with verdict | Settings |
|---|---|---|---|
| 0 | Lock | Status "guard" (the code tried to read test) → **BUG** | Not part of the ablation |
| 1 | G1 materiality | Pair-flip rate against the incumbent < 0.001 in ≥ majority of seeds → **INERT** | ER-1 `gate.pair_decomposition`, `materiality` |
| 2 | G2 matched baseline | (a) Spearman ≥ †0.999 to any registered baseline in ≥ majority of seeds; or (b)† gain over incumbent > 0 while gain over the strongest registered baseline ≤ 0 → **SCORE-MATCHED** | Spearman on the concatenated validation scores per seed. A NaN Spearman is not a match. A baseline with the same name as the candidate is excluded |
| — | Comparator† | — | The best of the incumbent and the registered baselines by seed-mean validation AUROC. Ties go to the incumbent |
| 3 | G3 benefit | Δ = seed-mean AUROC(candidate) − AUROC(comparator). If Δ ≤ 0: one LORD++ step with p = 1 → **NO EFFECT**. Otherwise: paired stratified bootstrap, B = †2,000, seed 0; SE_combined = √(SE²_boot + s²_seed/n); one-sided normal p; pass iff p ≤ α_t | ER-1 `paired_bootstrap`, `combined_interval`, `one_sided_p` (or the bit-identical vectorised version, §4.5) |
| 4 | G4 power† | Runs only if G3 fails with Δ > 0. Using ER-1 `benefit_verdict` on the **combined** 95% interval with band 0.001: NEUTRAL or HARMFUL → **NO EFFECT**; USEFUL or INCONCLUSIVE → **UNDERPOWERED**, logging MDD = (z₀.₉₅ + z₀.₈₀) × SE_combined ≈ 2.49 × SE | G4 is not in the ablation |
| 5 | G5 calibration | Only if the proposal returns probabilities: mean validation ECE (15 bins, **in-scope items only**) exceeds the plain model of the same arm (softmax for CE, Dirichlet mean for EDL) by > 0.01 → **MISCALIBRATES** | |
| 6 | G6 ceiling | Only for declared linear fusions (§3.1): AUROC exceeds the best in-sample linear fusion of the two parents (720 angles, validation arrays only) by > †0.0001 in any seed → **BUG** | ER-1 `fusion.linear_fusion`; validation arrays passed in both split slots |

**Registered baselines (G2), per cell, 8 scores:** energy and MSP of each arm; D_port of each arm (256-d bottleneck, PCA 128, whitened, tied covariance); D_full of each arm (768-d encoder output, no PCA, tied covariance). D_port and D_full use the negative marginal log-density under a uniform class prior.

EDL vacuity is a registered score that `PARENTS` may name, but it is not a G2 baseline. The ER-1 paper (§3.1, §6.1) shows vacuity is a strictly monotone function of clipped energy, and that EDL logits never reached the clip in any run. So vacuity is rank-identical to EDL energy, and G2(a) catches it through `energy_edl`. The observed Spearman is logged in scorecard pair 1.

**LORD++** (Ramdas et al., 2017):
- α = 0.1, initial wealth W₀ = †0.05.
- γ_j = 0.07720838 · log(max(j, 2)) / (j · e^{√log j}).
- α_t = γ_t·W₀ + (α − W₀)·γ_{t−τ₁} + α·Σ_{j≥2} γ_{t−τ_j}, where τ_j is the j-th rejection and missing terms are 0. Reject iff p_t ≤ α_t.
- One state per (rule variant, cell, ordering). Every proposal that reaches G3 consumes one step, including Δ ≤ 0 with p = 1. Proposals that fail G1 or G2 consume none.
- Reference values: α₁ = 0.002675838546; after a rejection at t = 1, α₂ = 0.003257748835.
- **Confirmed.** The α_t update matches LORD++ in Ramdas, Yang, Wainwright and Jordan 2017. The constant 0.07720838 is the onlineFDR default γ, and that series is not normalized to sum to 1, so the procedure is conservative relative to a γ that sums to 1. The reference values match the implementation: α₁ = 0.002675838546, and α₂ = 0.003257748835 after a rejection at t = 1.

**Leave-one-gate-out variants:** R2 without G1, G2, G3, G5 and G6, one at a time. Removing G3 means "accept if Δ > 0", with no LORD++ step and no G4. Removing any other gate skips its step. If G2 is removed, the comparator is the incumbent.

---

## 4. Statistical analysis

Everything in this section runs only after unblinding (§6), from the ledger, the decision files and the unsealed scores. No rule or gate is re-run after unblinding.

### 4.1 Truth

- **Test gain of a pair** (candidate, incumbent at acceptance, cell): a paired stratified bootstrap on test, B = 5,000, seed 0, resampling in-scope and OOS test items separately with one draw shared by both scores and all seeds. The gain is the seed mean of AUROC differences, and SE_combined = √(SE²_boot + s²_seed/n). Banking77-OOS uses the pooled 2,080 OOS test items; the ID-OOS and OOD-OOS subsets are a secondary breakdown only.
- **True improvement, primary:** the combined 95% interval of the test gain is entirely above 0 (strict inequality on the lower bound).
- **True improvement, strict:** primary holds, and in none of the other three cells where both scores exist does the same pair's combined 95% interval lie entirely below 0. A cell where either score is missing or not status-ok is skipped and counted.
- **Confirmed. Truth is relative to the incumbent at acceptance**, even when the critic's comparator was a baseline. This follows the design document; the README states it.

### 4.2 Metrics (per rule, cell and ordering)

- **FDP** = accepted changes that are not true improvements (primary) ÷ accepted changes. It is undefined when nothing is accepted, and is never written as 0. Strict FDP is reported alongside.
- **Stream truth set** S_cell = status-ok, non-canary proposals whose test gain over †**CE energy** has a combined 95% interval above 0. S_cell does not depend on ordering.
- **Recall. Confirmed.** Recall = |accepted proposals ∩ S_cell| ÷ |S_cell|. Numerator and denominator use the same CE-energy truth set, so recall ≤ 1. It is undefined if S_cell is empty.
- **Winner's-curse gap** = mean over accepted changes of (Δval − Δtest), both seed-mean AUROC differences against the incumbent at acceptance. Δval is recomputed from the validation arrays, not taken from rule records.
- **Aggregation over orderings:** the mean over orderings where the value is defined, with the number of excluded orderings reported. H1 is the exception: it is change-weighted, pooled over every accepted change in every ordering.
- **Final-pick quality:** for each rule, the last incumbent of each ordering is reported against CE energy and the **best single density chosen by validation AUROC** among D_port and D_full of each arm. Measures: test AUROC (seed mean), FPR@95 and ECE, given as median, IQR and range over orderings. **Confirmed.** ECE is the proposal's own in-scope test ECE if it returns probabilities, otherwise the plain-model ECE of its arm.

### 4.3 Intervals

Two-level bootstrap, 2,000 replicates, seed 0, percentile intervals. Each replicate:
1. Resamples orderings with replacement, using the same indices for every rule and cell.
2. Draws proposal multiplicity weights from a multinomial over the proposals, shared across cells.
3. Recomputes each statistic with every accepted change weighted by its candidate's weight; a weight of 0 drops the change.

**Confirmed.** This approximates resampling proposals without replaying the rules. Replaying was judged infeasible for R2 within the analysis window. Holm is applied across H1–H4 as in §1.

### 4.4 Gate ablation

For each leave-one-gate-out variant, per cell, the table reports:
- the mean over orderings of n_false(variant) − n_false(R2), paired within ordering;
- the same for FDP;
- the count of "slipped" false accepts, meaning those whose (candidate, incumbent) pair full R2 did not accept in that ordering.

Intervals use §4.3. The gate ablation is descriptive and outside the Holm family.

### 4.5 Computation

The paired bootstrap on test and validation uses a vectorised re-implementation of ER-1's `paired_bootstrap`. It is **bit-identical** to ER-1's function on every equivalence fixture (maximum absolute difference 0.0) and about 150× faster. A runtime self-check compares the two at B = 20 on the first pair in each process. Statistics are cached by array-content hash of (candidate, comparator or incumbent, cell, B, seed, ER-1 commit).

### 4.6 Rediscovery scorecard

The scorecard holds eight fixed (candidate, incumbent, cell) triples. Each runs through the frozen critic with a fresh LORD++ state, in the listed order. It is separate from the FDP measurement, uses no orderings, and is descriptive.

**Scorecard mode. Confirmed.** The paper's findings are pairwise claims ("X beats Y"). Each pair is graded strict and lenient. Scorecard pairs run with:
- G2(a) (rank match) against the registered baselines **excluding the candidate's own score family in both arms**, because the paper shows CE and EDL densities tie, so the twin would always match;
- **G2(b) off and the comparator equal to the incumbent**;
- every other gate unchanged.

Under the stream rules instead, pairs 5 and 7 would be SCORE-MATCHED against stronger baselines (CE energy on CLINC150, D_port on Banking77-OOS). That says something about the critic's question, not about the paper's finding. Stream-mode verdicts for the eight pairs are also reported, because the same configurations sit in the templated stream.

**Grading.** Each pair has a strict expected verdict and a lenient set. Report "strict matches / 8" and "lenient matches / 8". The expectations come from the paper's test-set results; the critic sees validation only, so several gains are smaller than validation can resolve (paper §5.6: AUROC SE ≈ 0.011 on CLINC150 validation, ≈ 0.007 on Banking77-OOS validation).

| # | Cell | Candidate | Incumbent | Paper evidence (test) | Strict | Lenient |
|---|---|---|---|---|---|---|
| 1 | CLINC150/BERT | EDL vacuity | CE MSP | Vacuity − CE MSP = +0.0101, entirely a scoring-rule effect; vacuity is rank-identical to energy | SCORE-MATCHED | SCORE-MATCHED |
| 2 | CLINC150/BERT | EDL D_port, PCA 128, whitened | EDL D_port, PCA 128, not whitened | Whitening is an exact no-op for tied Mahalanobis (§3.4) | INERT | INERT |
| 3 | CLINC150/BERT | qsigmoid × mul on EDL D_port (coupled vacuity) | EDL vacuity | −0.0047, HARMFUL in 15/15 seed × normalizer cells; ECE 0.011 → 0.159 | NO EFFECT | NO EFFECT, MISCALIBRATES |
| 4 | CLINC150/BERT | qsigmoid × mul on EDL D_port | EDL D_port alone | Coupling within 0.001 AUROC of the density in 6 of 8 study × density combinations | INERT | INERT, SCORE-MATCHED |
| 5 | CLINC150/BERT | EDL bottleneck density, no PCA | EDL D_port (PCA 128) | +0.0026 | ACCEPT | ACCEPT, UNDERPOWERED |
| 6 | Banking77-OOS/BERT | EDL D_port | EDL vacuity | 0.9425 vs 0.9310 = +0.0115 | ACCEPT | ACCEPT, UNDERPOWERED |
| 7 | Banking77-OOS/BERT | ecdf, add_log, λ = 5 on EDL D_full | EDL vacuity (the better of its two inputs) | +0.0035 [+0.0005, +0.0066] over vacuity; +0.0096 over D_full | ACCEPT | ACCEPT, UNDERPOWERED |
| 8 | CLINC150/BERT | Relative Mahalanobis on the EDL encoder output | EDL D_full | −0.0037 | NO EFFECT | NO EFFECT |

Pre-declared predictions:
- Pair 5 is likely UNDERPOWERED, because the CLINC150 validation set has only 100 OOS items.
- Pair 7 is likely UNDERPOWERED: its test interval is narrow and validation has about a third of test's OOS items.
- For pair 4, the paper establishes an AUROC difference within 0.001, which does not by itself imply a pair-flip rate below 0.001. INERT is the strict expectation; SCORE-MATCHED is lenient.

Any finding the critic accepts that is not in the paper is checked on test and in the other three cells after unblinding. It is reported as a candidate finding, not a result.

### 4.7 Experiment 2 outcomes (secondary, descriptive)

**Design.**
- Arms: A = R0 with the proposer seeing its own kept and discarded history; B = R2 with shared memory; C = R2 without memory.
- Benchmarks: CLINC150 and Banking77-OOS, †BERT only.
- 3 runs per arm and benchmark, about 80 proposals each.
- Each run has its own LORD++ state and memory namespace, and arm B starts empty.

**Wasted proposal†:** a rejection whose (FAMILY, verdict) pair matches an earlier rejection in the same run.

**Reported:**
- H5 per run, with the mean per arm;
- after unblinding, FDP and recall per arm (truth as in §4.1);
- the final pick's test AUROC, FPR@95 and ECE against CE energy and the best validation-chosen density.

No intervals are claimed. If not started by Sun 04:00 PT, H5 is not tested and is reported as not run.

---

## 5. Operational defaults and fallbacks

| Item | Registered value |
|---|---|
| Templated proposals | †300, `default_rng(0)`, plus 10 scorecard files |
| LLM stream | 181 proposals, frozen in §3.0; Sat 23:00 gate (≥ 100 distinct) met |
| Scout cadence | †one Exa search per 10 proposals, ≤ 3 cards |
| Analyst cadence | †one lesson per 25 proposals, citing proposal hashes |
| Sandbox limit | †60 s per seed per cell; no network; open volume read-only |
| Item-wise check | †random 10% subset, values equal to 1e-6 |
| Orderings | †200 per cell, seed 0; †fallback 100 if not finished by Sun 05:00 |
| Critic bootstrap | †B = 2,000, seed 0 |
| Truth bootstrap | B = 5,000, seed 0 |
| Hypothesis bootstrap | 2,000 replicates, seed 0 |
| Fine-tune fallback | †laptop MPS caches if the 40 CUDA caches are late; disclosed |
| Testbed B go/no-go | Sat 22:00 deadline missed; run late as an exploratory pilot (§2.2, §7) |
| Testbed B held-out sizes | †4,000 / 4,000 / 2,000 |
| Testbed B kNN, layers | †k = 10; †layers 12, 16, 20 |
| Memory | Never contains a test number; a local JSON fallback behind the same interface |
| Secrets | Modal secrets and a git-ignored `.env`; never in prompts, ledger rows or memory |

---

## 6. Freeze and unblinding procedure

**Timing statement.** This fill was written at **2026-11-15 09:40 UTC**, after the planned Sat 19:00 PT. The timestamp of record is the author date of the commit that registers this file. State at that commit:

| Step | State |
|---|---|
| Proposal generation | Done: 500 proposals (§3.0) |
| Validation scoring | Done: 1,996 ok rows, 5 seeds each |
| Experiment 1 decisions | **None.** `runs/decisions/` is empty; the replay refuses to run while this file is a draft |
| Experiment 2 (closed loop) | **Not started.** It starts only after this commit, because its proposer reacts to verdicts |
| Test scores | **Not computed** for any proposal (`not_run`) |
| Baseline sanity read on test (§2.1) | **Not done.** There is no `runs/guard.log` |
| Testbed B | Not started; no held-out shift read |

**Validation data seen before registration.** The thresholds, gates and truth definitions marked † come from the design document, written before the ledger existed. The confirmed choices were drafted from the ER-1 paper and the R1/R2 specs. Exposure while preparing this file: one ledger row's validation AUROCs were printed, proposal `004d96ef1bc0` on `banking77_oos/bert`, five seeds 0.907546, 0.900863, 0.915779, 0.903491, 0.908495. The rest of the ledger inspection was counts, statuses, stream tags and hashes. No threshold in this file was chosen by looking at the ledger.

Anything that ran before this commit and depends on a setting above is listed in §7.

**Decision-log hash.** When Experiment 1 decisions exist for every rule, variant, ordering and cell, `make seal-hash` computes:

```
H = SHA-256 over the concatenation, for each file f in runs/decisions/ in bytewise-sorted
    POSIX relative-path order, of:  relpath(f) + "\n" + hex(SHA-256(bytes(f))) + "\n"
```

It writes the hex digest to `runs/DECISIONS.sha256`. The ledger is in git already and is timestamped by its commits.

**Unblinding, in order** (planned Sun 07:00 PT):
1. Check that decisions exist for every rule, variant, ordering and cell, and that the count of orderings is 200, or 100 under the fallback.
2. `make seal-hash`, then commit and push `runs/DECISIONS.sha256`. The commit time is the public proof that every decision predates any test read.
3. `make unblind`: `store.unseal()` recomputes H and refuses to open the sealed volume if the file is missing or differs.
4. Truth, FDP, recall, the ablation, the scorecard and H1–H4 are computed in one pass by `make report`. No rule or gate is re-run afterwards.

**Verification by a reader:**
- the commit adding this file predates the first decision file;
- the commit adding `DECISIONS.sha256` predates every table and figure;
- the hash matches the decision files;
- `make reproduce` rebuilds every table from the ledger, the decisions and the released sealed scores.

**Responsible for unblinding:** Ryu (@mid9t).

**After unblinding.** A bug found in analysis code (not in a rule or gate) is fixed and stated in the README, and the original output is kept in history. A bug found in a rule or gate makes the corrected result exploratory, and the original is still reported.

---

## 7. Deviations

**Changes from the design document made in this registration** (made before any test number was seen; listed for transparency):

| # | Design document | Registered here | Reason |
|---|---|---|---|
| 1 | Templated couplings declare PARENTS, so G6 applies | PARENTS only for linear fusions; G6 applies only to them | G6 is a linear ceiling; a nonlinear coupling can exceed it without a bug (ER-1 paper §3.3) |
| 2 | Scorecard pairs run like stream proposals | Scorecard mode (§4.6): family-twin exclusion in G2(a), G2(b) off, comparator = incumbent; stream-mode verdicts also reported | The paper's findings are pairwise; under stream rules, pairs 5 and 7 are decided by unrelated stronger baselines |
| 3 | Scorecard expects ACCEPT for pairs 5–7 | Strict and lenient sets, with UNDERPOWERED predicted for pairs 5 and 7 | Validation power (paper §5.6) |
| 4 | Recall numerator "true improvements accepted" | Accepted ∩ CE-energy truth set | Keeps recall ≤ 1 |
| 5 | ER-1 `paired_bootstrap` | Bit-identical vectorised version | Compute window; equivalence tested |
| 6 | Single p-values for H2 and H3 not specified | §1 | Needed for Holm |
| 7 | Pre-registration committed Sat 19:00, before the stream is built | Committed Sun 15 Nov, after 09:40 UTC (the commit author date is the record), after proposal generation and validation scoring, before any decision or test read; stream frozen by hash (§3.0) | Schedule slip. The replay design keeps decisions after registration |
| 8 | 200–400 LLM proposals by Sun morning | 181, frozen | Schedule; reported with the stream breakdown |
| 9 | Drop Testbed B if not cached by Sat 22:00 | Run late as an exploratory pilot (§2.2) | Time became available; labelled exploratory |
| 10 | 300 + 8 templated proposals | 310 files (§3.0) | The seed-0 draw of 300 missed every scorecard configuration. The eight findings are pairs and use 10 distinct files, so 300 + 10 = 310 |

**Later deviations:** append one line per change, with date, time, what changed and why. Every analysis outside this plan is labelled exploratory.

- Score-file manifest (`runs/score_manifest.sha256`): its commit hash is written on this line when that file is committed, before the first decision file.
- 2026-11-15 20:14 PT. Confirmatory tables and H1–H4 use orderings 0–99. All 200 orderings are published and copied to `tables/exploratory/`. The replay finished after the Sunday 05:00 PT fallback, so 200 orderings were available.
- 2026-11-15 20:14 PT. Final picks that return probabilities are reported with the arm's plain-model test ECE. The sealer stores scores and does not store the proposal's own probabilities.
