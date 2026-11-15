import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and pooled-space Mahalanobis distance capture complementary logit uncertainty and representation novelty. Train-fitted empirical percentile maps place both signals on a common scale before combining them."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    energy_ref = np.sort(train_energy)
    dist_ref = np.sort(train_dist)
    n = float(len(tr.labels))

    energy_q = np.searchsorted(energy_ref, eval_energy, side="right") / n
    dist_q = np.searchsorted(dist_ref, eval_dist, side="right") / n

    return 0.5 * (energy_q + dist_q)
