import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Training empirical CDFs place logit uncertainty and pooled-space class distance on a shared in-scope tail scale. Taking the larger calibrated anomaly score detects queries that are unsupported either by classifier evidence or by intent geometry."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0101", "m_0105"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    energy_order = np.sort(train_energy)
    dist_order = np.sort(train_dist)
    n = float(len(tr.labels))

    energy_tail = np.searchsorted(energy_order, eval_energy, side="right") / n
    dist_tail = np.searchsorted(dist_order, eval_dist, side="right") / n

    return np.maximum(energy_tail, dist_tail)
