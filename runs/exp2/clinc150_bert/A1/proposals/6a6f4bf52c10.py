import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and pooled-space Mahalanobis novelty capture complementary classifier uncertainty and representation mismatch. Train empirical CDF calibration puts their upper-tail evidence on a shared scale before combining them."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0039"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = ctx.h.energy(tr.logits)
    train_distance = g.min_mahalanobis_sq(tr.pooled)

    energy_sorted = np.sort(train_energy)
    distance_sorted = np.sort(train_distance)
    n = float(len(tr.labels))

    eval_energy = ctx.h.energy(ev.logits)
    eval_distance = g.min_mahalanobis_sq(ev.pooled)

    energy_cdf = np.searchsorted(energy_sorted, eval_energy, side="right") / n
    distance_cdf = np.searchsorted(distance_sorted, eval_distance, side="right") / n

    energy_cdf = np.clip(energy_cdf, 0.0, 1.0 - 1e-12)
    distance_cdf = np.clip(distance_cdf, 0.0, 1.0 - 1e-12)

    return -np.log1p(-energy_cdf) - np.log1p(-distance_cdf)
