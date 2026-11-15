import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and pooled-feature Mahalanobis distance capture complementary classifier uncertainty and representation mismatch. Empirical train-distribution calibration maps both signals to a common percentile scale before fusion."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0002", "m_0004"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    energy_ref = np.sort(train_energy)
    dist_ref = np.sort(train_dist)
    n = float(len(tr.labels))

    energy_pct = np.searchsorted(energy_ref, eval_energy, side="right") / n
    dist_pct = np.searchsorted(dist_ref, eval_dist, side="right") / n

    return energy_pct + dist_pct
