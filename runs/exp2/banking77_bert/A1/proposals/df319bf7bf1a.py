import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and encoder-space Mahalanobis distance capture complementary classifier uncertainty and representation novelty. Train-fitted quantile-sigmoid calibration places both signals on a common bounded scale without using evaluation statistics."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0001"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    energy_inlier = ctx.h.normalizer("qsigmoid", -train_energy)
    dist_inlier = ctx.h.normalizer("qsigmoid", -train_dist)

    energy_oos = 1.0 - energy_inlier(-eval_energy)
    dist_oos = 1.0 - dist_inlier(-eval_dist)
    return np.asarray(0.5 * energy_oos + 0.5 * dist_oos, dtype=float)
