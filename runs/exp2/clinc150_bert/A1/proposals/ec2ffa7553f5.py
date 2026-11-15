import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "MSP captures classifier uncertainty while pooled-space Mahalanobis distance captures representation novelty. Empirical train CDF calibration puts both signals on a comparable tail-probability scale before combining them."
SOURCE = ""
PARENTS = ["msp", "d_port"]
MEMORY_USED = ["m_0015", "m_0019"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    tr_msp = np.asarray(ctx.h.msp(tr.logits), dtype=float)
    ev_msp = np.asarray(ctx.h.msp(ev.logits), dtype=float)
    tr_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    ev_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def tail_cdf(train_values, values):
        ref = np.sort(np.nan_to_num(train_values, nan=0.0, posinf=1e12, neginf=-1e12))
        x = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        return np.searchsorted(ref, x, side="right") / float(ref.size + 1)

    msp_tail = tail_cdf(tr_msp, ev_msp)
    dist_tail = tail_cdf(tr_dist, ev_dist)

    eps = 1e-6
    return -0.5 * (np.log1p(-np.clip(msp_tail, 0.0, 1.0 - eps)) +
                   np.log1p(-np.clip(dist_tail, 0.0, 1.0 - eps)))
