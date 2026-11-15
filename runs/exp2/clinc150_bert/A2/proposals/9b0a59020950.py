import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "MSP uncertainty and pooled-space Mahalanobis distance capture complementary failure modes: an OOS query may be classifier-uncertain, geometrically atypical, or both. Training empirical percentiles put both signals on a common tail-probability scale, and their product emphasizes jointly unsupported inputs."
SOURCE = ""
PARENTS = ["msp", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_logit = np.asarray(ctx.h.msp(tr.logits), dtype=float)
    eval_logit = np.asarray(ctx.h.msp(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def tail_probability(reference, values):
        reference = np.asarray(reference, dtype=float)
        values = np.asarray(values, dtype=float)
        finite_ref = reference[np.isfinite(reference)]
        if finite_ref.size == 0:
            finite_ref = np.array([0.0, 1.0])
        finite_values = np.nan_to_num(
            values,
            nan=float(np.median(finite_ref)),
            posinf=float(np.max(finite_ref)),
            neginf=float(np.min(finite_ref)),
        )
        ordered = np.sort(finite_ref)
        return (np.searchsorted(ordered, finite_values, side="right") + 1.0) / (
            ordered.size + 2.0
        )

    p_logit = tail_probability(train_logit, eval_logit)
    p_dist = tail_probability(train_dist, eval_dist)

    return np.sqrt(p_logit * p_dist)
