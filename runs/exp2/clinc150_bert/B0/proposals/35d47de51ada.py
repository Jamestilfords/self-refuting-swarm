import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy uncertainty and pooled-feature Mahalanobis distance are converted to train-calibrated empirical tail scales, then combined with a harmonic mean. This conjunction emphasizes queries that are simultaneously classifier-uncertain and poorly supported by the in-scope feature distribution."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_logit = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_logit = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    logit_ref = np.sort(train_logit)
    dist_ref = np.sort(train_dist)
    n = float(len(tr.labels))

    logit_tail = np.searchsorted(logit_ref, eval_logit, side="right") / n
    dist_tail = np.searchsorted(dist_ref, eval_dist, side="right") / n

    eps = 1e-6
    return 2.0 * logit_tail * dist_tail / (logit_tail + dist_tail + eps)
