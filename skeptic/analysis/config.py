"""Every number and choice that defines Experiment 1 truth.

`python -m skeptic.analysis.config --markdown` prints the table that goes into
PREREGISTRATION.md §4. Values marked with a dagger in that table are the
proposed defaults.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

ER1_COMMIT = "e877e9159d"

_RULE_NAMES = MappingProxyType({
    "ratchet": "R0",
    "thresholdout": "R1",
    "critic": "R2",
})


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class AnalysisConfig:
    """Pre-registered analysis choices. The digest is written into every CSV."""

    cells: tuple[str, ...] = (
        "clinc150/bert",
        "clinc150/roberta",
        "banking77_oos/bert",
        "banking77_oos/roberta",
    )
    test_B: int = 5000
    test_seed: int = 0
    truth_ci_level: float = 0.95
    recall_reference: str = "energy_ce"  # †
    recall_numerator: str = "stream_truth_set"  # † D1
    strict_reversal: str = "ci95_high_below_0"
    hyp_B: int = 2000
    hyp_seed: int = 0
    hyp_resample: str = "orderings_and_proposal_weights"  # † D3
    holm_alpha: float = 0.05
    h1_cells: tuple[str, ...] = ("clinc150/bert", "clinc150/roberta")
    h2_min_cells: int = 3
    h3_cells: tuple[str, ...] = ("banking77_oos/bert", "banking77_oos/roberta")
    h3_threshold: float = 0.5
    h4_small: tuple[str, ...] = ("clinc150/bert", "clinc150/roberta")
    h4_large: tuple[str, ...] = ("banking77_oos/bert", "banking77_oos/roberta")
    final_pick_ece: str = "proposal_probs_else_plain_model"  # † D5
    rule_names: Mapping[str, str] = _RULE_NAMES
    gate_variants: tuple[str, ...] = (
        "R2-noG1",
        "R2-noG2",
        "R2-noG3",
        "R2-noG5",
        "R2-noG6",
    )
    registered_orderings: int = 100
    density_names: tuple[str, ...] = (
        "d_port_ce",
        "d_full_ce",
        "d_port_edl",
        "d_full_edl",
    )
    ratchet_rule: str = "R0"
    critic_rule: str = "R2"

    def digest(self) -> str:
        payload = {
            field.name: _plain(getattr(self, field.name))
            for field in dataclasses.fields(self)
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    def canonical_rule(self, directory: str) -> str:
        return str(self.rule_names.get(directory, directory))


DEFAULT = AnalysisConfig()


def definitions_markdown(config: AnalysisConfig = DEFAULT) -> str:
    """Definitions PREREGISTRATION.md §4 quotes, plus the config values."""
    rows = [
        ("cells", ", ".join(config.cells), "Four benchmark × backbone cells."),
        ("test_B", str(config.test_B), "Paired bootstrap replicates for a truth interval."),
        ("test_seed", str(config.test_seed), "Seed for every truth bootstrap."),
        ("truth_ci_level", str(config.truth_ci_level), "Primary truth uses the combined 95% interval."),
        ("recall_reference †", config.recall_reference, "S_cell is truth against this registered baseline."),
        ("recall_numerator †", config.recall_numerator, "D1. Numerator is membership in S_cell."),
        ("strict_reversal", config.strict_reversal, "D6. A reversal is a 95% interval entirely below 0."),
        ("hyp_B", str(config.hyp_B), "Two-level bootstrap replicates for H1–H4 and the gate ablation."),
        ("hyp_seed", str(config.hyp_seed), "Seed for the two-level bootstrap."),
        ("hyp_resample †", config.hyp_resample, "D3. Resample orderings and proposal weights. Do not replay rules."),
        ("holm_alpha", str(config.holm_alpha), "Family-wise level for the four hypotheses."),
        ("h1_cells", ", ".join(config.h1_cells), "Cells pooled for the winner's-curse hypothesis."),
        ("h2_min_cells", str(config.h2_min_cells), "H2 is supported only if at least this many cells clear zero."),
        ("h3_cells", ", ".join(config.h3_cells), "Cells in the Banking77 recall ratio."),
        ("h3_threshold", str(config.h3_threshold), "D4. H3 tests the ratio against this floor."),
        ("h4_small", ", ".join(config.h4_small), "CLINC150 cells in the false-discovery contrast."),
        ("h4_large", ", ".join(config.h4_large), "Banking77 cells in the false-discovery contrast."),
        ("final_pick_ece †", config.final_pick_ece, "D5. Proposal probabilities, otherwise the arm's plain-model ECE."),
        ("rule_names", ", ".join(f"{src}→{dst}" for src, dst in config.rule_names.items()), "Decision directory names."),
        ("gate_variants", ", ".join(config.gate_variants), "Leave-one-gate-out directories. G4 has no variant."),
        ("registered_orderings", str(config.registered_orderings),
         "Confirmatory tables use orderings below this index. The rest are exploratory."),
        ("density_names", ", ".join(config.density_names), "Best single density is chosen on validation AUROC."),
        ("ratchet_rule", config.ratchet_rule, "Canonical name of the keep-if-better rule."),
        ("critic_rule", config.critic_rule, "Canonical name of the full critic."),
    ]
    lines = [
        "| Choice | Value | Role |",
        "|---|---|---|",
    ]
    for name, value, role in rows:
        lines.append(f"| `{name}` | {value} | {role} |")
    lines.append("")
    lines.append("### Definitions")
    lines.append("")
    lines.append(
        "A change is a **true improvement** when its combined 95% interval for "
        "test AUROC(candidate) − AUROC(incumbent at acceptance) lies strictly above 0. "
        "A lower bound of exactly 0 is not a true improvement. The interval adds the "
        "utterance-bootstrap standard error to the seed-level standard error. "
        "Strict truth additionally requires that the same pair is not reversed in any "
        "other cell whose scores exist. A cell is reversed only when its 95% interval "
        "lies entirely below 0. A missing or non-ok score is skipped and reported; "
        "it is not a reversal. Banking77 truth uses the pooled OOS test set. "
        "The ID-OOS and OOD-OOS breakdown is secondary and does not enter FDP or recall."
    )
    lines.append("")
    lines.append(
        "**False discovery proportion** (one rule, cell, and ordering) is the number of "
        "accepted changes that are not true improvements divided by the number of accepted "
        "changes. Truth is the gain over the incumbent at acceptance, including when the "
        "critic compared the proposal with a baseline instead of that incumbent. "
        "Primary truth is the pre-registered metric. Strict FDP is reported beside it. "
        "FDP is undefined when the rule accepted nothing, and undefined is not written as zero."
    )
    lines.append("")
    lines.append(
        "**Recall** (one rule, cell, and ordering) is the number of accepted proposals "
        "that belong to S_cell, divided by the size of S_cell. S_cell is the set of "
        "status-ok, non-canary proposals whose test gain over CE energy has a combined "
        "95% interval strictly above 0. The numerator is membership in that same set, "
        "so recall cannot exceed 1. Recall is undefined when S_cell is empty."
    )
    lines.append("")
    lines.append(
        "**Winner's-curse gap** is the mean, over accepted changes, of validation gain "
        "minus test gain. Both gains are against the incumbent at acceptance. "
        "Validation gain is recomputed from the open arrays. "
        "The H1 aggregate pools changes across orderings; it is not the mean of per-ordering means."
    )
    lines.append("")
    lines.append(
        "**Aggregation across orderings** is the mean of the per-ordering values on the "
        "orderings where the value is defined. FDP is averaged over orderings with at least "
        "one acceptance. The number of excluded orderings is reported. "
        "With a stream tag (`templated` or `llm`), only candidates carrying that tag count, "
        "and S_cell is restricted the same way. Pooled results are reported as well."
    )
    lines.append("")
    lines.append(
        "**Final-pick quality** scores the last accepted proposal in each ordering "
        "(CE energy when nothing was accepted) by its seed-mean test AUROC, seed-mean "
        "FPR at 95% OOS recall, and in-scope ECE. ECE uses the proposal's stored in-scope "
        "test probabilities when they exist, otherwise the plain-model ECE of its arm. "
        "The table reports the median, interquartile range, minimum, and maximum over "
        "orderings, next to CE energy and the best single density. That density is the "
        "best of d_port_ce, d_full_ce, d_port_edl, and d_full_edl by validation AUROC, never by test."
    )
    lines.append("")
    lines.append(
        "**Gate ablation** compares each leave-one-gate-out variant with the full critic "
        "on the same ordering. Slipped changes are false accepts in the variant whose "
        "candidate and incumbent the full critic did not accept in that ordering. "
        "The reported contrasts are the mean paired differences of the false-accept count "
        "and of FDP, with a two-level bootstrap interval on the false-accept contrast."
    )
    lines.append("")
    lines.append(
        "**H1.** Mean of (validation gain − test gain) over every change the ratchet "
        "accepted in every ordering, pooling both CLINC150 cells. Supported when the "
        "95% bootstrap interval lies above 0 and the Holm-adjusted two-sided p-value "
        "is at most 0.05. Two-sided p = max(2·min(P*(stat ≤ 0), P*(stat ≥ 0)), 1/B)."
    )
    lines.append("")
    lines.append(
        "**H2.** In each cell, the mean over orderings of FDP(ratchet) − FDP(critic), "
        "using only orderings where both proportions are defined. An ordering where either "
        "rule accepted nothing is excluded and counted. Supported when the 95% interval "
        "lies above 0 in at least 3 of the 4 cells and the Holm-adjusted p-value is at "
        "most 0.05. The single p-value is the third-smallest of the four per-cell "
        "two-sided p-values."
    )
    lines.append("")
    lines.append(
        "**H3.** Mean recall of the critic divided by mean recall of the ratchet, over "
        "the cell-ordering pairs in both Banking77 cells. Supported when the point "
        "estimate is at least 0.5. The interval and the Holm-adjusted p-value are "
        "reported either way. Two-sided p = max(2·min(P*(ratio ≤ 0.5), P*(ratio ≥ 0.5)), 1/B)."
    )
    lines.append("")
    lines.append(
        "**H4.** Mean ratchet FDP on the CLINC150 cells minus mean ratchet FDP on the "
        "Banking77 cells. Supported when the 95% interval lies above 0 and the "
        "Holm-adjusted two-sided p-value is at most 0.05."
    )
    lines.append("")
    lines.append(
        "**Two-level bootstrap.** For each replicate, draw the observed number of ordering "
        "indices with replacement (200, or 100 under the pre-declared fallback) and reuse "
        "those indices for every rule and cell. Then draw proposal multiplicity weights "
        "from a uniform multinomial, shared across cells. Recompute each statistic with "
        "accepted changes weighted by the candidate's multiplicity. Weight 0 drops a change. "
        "This does not replay the rules. Intervals are percentile intervals. "
        f"Holm adjustment is then applied to the four p-values at alpha {config.holm_alpha}."
    )
    lines.append("")
    lines.append(
        f"Config digest: `{config.digest()}`. ER-1 commit: `{ER1_COMMIT}`."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print the pre-registered truth definitions.")
    parser.add_argument("--markdown", action="store_true", help="write the §4 definitions table")
    args = parser.parse_args(argv)
    if not args.markdown:
        parser.error("pass --markdown")
    print(definitions_markdown(), end="")


if __name__ == "__main__":
    main()
