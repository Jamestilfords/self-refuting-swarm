import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Train empirical-CDF calibration maps MSP uncertainty and pooled-space Mahalanobis distance onto a shared outlier scale, while their interaction emphasizes examples that are simultaneously uncertain and representation-shifted."
SOURCE = ""
PARENTS = ["msp", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_logit = ctx.h.msp(tr.logits)
    eval_logit = ctx.h.msp(ev.logits)
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    def ecdf(train_values, values):
        reference = np.sort(np.asarray(train_values, dtype=float))
        reference = np.nan_to_num(
            reference, nan=0.0, posinf=1e12, neginf=-1e12
        )
        query = np.nan_to_num(
            np.asarray(values, dtype=float), nan=0.0, posinf=1e12, neginf=-1e12
        )
        return np.searchsorted(reference, query, side="right") / float(
            max(reference.size, 1)
        )

    logit_u = ecdf(train_logit, eval_logit)
    dist_u = ecdf(train_dist, eval_dist)

    combined = logit_u + dist_u + 0.75 * logit_u * dist_u
    return np.nan_to_num(combined, nan=0.0, posinf=3.0, neginf=0.0)
