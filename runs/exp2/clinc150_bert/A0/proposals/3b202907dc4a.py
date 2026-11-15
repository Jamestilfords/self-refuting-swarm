import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "MSP uncertainty and pooled-space Mahalanobis distance capture complementary failures: unsupported queries can be uncertain, geometrically distant, or both. Train empirical-CDF calibration places both signals on a shared in-distribution tail scale before fusion."
SOURCE = ""
PARENTS = ["msp", "d_full"]
MEMORY_USED = ["m_0059"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    tr_logit = ctx.h.msp(tr.logits)
    ev_logit = ctx.h.msp(ev.logits)
    tr_dist = g.min_mahalanobis_sq(tr.pooled)
    ev_dist = g.min_mahalanobis_sq(ev.pooled)

    logit_ref = np.sort(np.asarray(tr_logit, dtype=float))
    dist_ref = np.sort(np.asarray(tr_dist, dtype=float))
    n = float(len(tr.labels))

    logit_tail = np.searchsorted(logit_ref, ev_logit, side="right") / n
    dist_tail = np.searchsorted(dist_ref, ev_dist, side="right") / n

    eps = 1e-6
    return -np.log1p(-np.clip(logit_tail, 0.0, 1.0 - eps)) + \
           -np.log1p(-np.clip(dist_tail, 0.0, 1.0 - eps))
