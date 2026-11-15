import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Classifier entropy and pooled-space Mahalanobis novelty capture complementary OOS signals. "
    "Mapping each to its train empirical tail probability makes their consensus directly comparable without using eval-set statistics."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0099"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_logit = np.nan_to_num(
        ctx.h.entropy(tr.logits), nan=0.0, posinf=1e12, neginf=-1e12
    )
    eval_logit = np.nan_to_num(
        ctx.h.entropy(ev.logits), nan=0.0, posinf=1e12, neginf=-1e12
    )

    train_dist = np.nan_to_num(
        g.min_mahalanobis_sq(tr.pooled), nan=0.0, posinf=1e12, neginf=-1e12
    )
    eval_dist = np.nan_to_num(
        g.min_mahalanobis_sq(ev.pooled), nan=0.0, posinf=1e12, neginf=-1e12
    )

    logit_reference = np.sort(train_logit)
    dist_reference = np.sort(train_dist)
    n = float(len(tr.labels))

    logit_tail = np.searchsorted(logit_reference, eval_logit, side="right") / n
    dist_tail = np.searchsorted(dist_reference, eval_dist, side="right") / n

    return 0.5 * logit_tail + 0.5 * dist_tail
