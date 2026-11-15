import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "MSP uncertainty and class-conditional encoder distance are converted to training-set percentile scales before a harmonic fusion. The harmonic mean emphasizes queries that are simultaneously uncertain in logit space and atypical in representation space."
SOURCE = ""
PARENTS = ["msp", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_msp = np.asarray(ctx.h.msp(tr.logits), dtype=float)
    eval_msp = np.asarray(ctx.h.msp(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    msp_ref = np.sort(train_msp)
    dist_ref = np.sort(train_dist)
    n_msp = max(len(msp_ref), 1)
    n_dist = max(len(dist_ref), 1)

    msp_q = np.searchsorted(msp_ref, eval_msp, side="right") / n_msp
    dist_q = np.searchsorted(dist_ref, eval_dist, side="right") / n_dist

    msp_q = np.clip(msp_q, 1e-4, 1.0)
    dist_q = np.clip(dist_q, 1e-4, 1.0)
    return 2.0 * msp_q * dist_q / (msp_q + dist_q)
