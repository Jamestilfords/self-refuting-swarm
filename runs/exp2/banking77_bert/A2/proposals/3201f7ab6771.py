import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "MSP captures classifier uncertainty while full-space Mahalanobis distance captures representation novelty. Train-derived percentile calibration makes their scales comparable, and a noisy-OR fusion emphasizes items anomalous by either signal."
SOURCE = ""
PARENTS = ["msp", "d_full"]
MEMORY_USED = ["m_0003", "m_0015"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_msp = ctx.h.msp(tr.logits)
    eval_msp = ctx.h.msp(ev.logits)
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    ref_msp = np.sort(train_msp)
    ref_dist = np.sort(train_dist)

    q_msp = np.searchsorted(ref_msp, eval_msp, side="right") / float(len(ref_msp))
    q_dist = np.searchsorted(ref_dist, eval_dist, side="right") / float(len(ref_dist))

    q_msp = np.clip(q_msp, 1e-6, 1.0 - 1e-6)
    q_dist = np.clip(q_dist, 1e-6, 1.0 - 1e-6)
    return q_msp + q_dist - q_msp * q_dist
