"""The only module that imports ER-1. Everything else uses these names."""

from __future__ import annotations

from bert_daedl.losses.edl import logits_to_alpha
from bert_daedl.posthoc.cache import find_runs, load_cache, load_runs
from bert_daedl.posthoc.fusion import linear_fusion
from bert_daedl.posthoc.gate import benefit_verdict, materiality, pair_decomposition
from bert_daedl.posthoc.scoring import (
    dirichlet_ece,
    ece_from_probs,
    logit_scores,
    logp_from_run,
    softmax_ece,
    spearman,
)
from bert_daedl.posthoc.stats import (
    auroc_scores,
    combined_interval,
    fpr_at_95,
    holm,
    one_sided_p,
    paired_bootstrap,
)

ER1_COMMIT = "e877e9159d"

__all__ = [
    "ER1_COMMIT",
    "auroc_scores",
    "benefit_verdict",
    "combined_interval",
    "dirichlet_ece",
    "ece_from_probs",
    "find_runs",
    "fpr_at_95",
    "holm",
    "linear_fusion",
    "load_cache",
    "load_runs",
    "logit_scores",
    "logits_to_alpha",
    "logp_from_run",
    "materiality",
    "one_sided_p",
    "pair_decomposition",
    "paired_bootstrap",
    "softmax_ece",
    "spearman",
]
