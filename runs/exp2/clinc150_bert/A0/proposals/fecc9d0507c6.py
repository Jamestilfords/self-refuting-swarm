import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and bottleneck-space relative Mahalanobis distance capture complementary classifier uncertainty and representation novelty. Train-only ECDF calibration maps both signals to a common upper-tail scale before combining them."
SOURCE = ""
PARENTS = ["energy", "d_port"]
MEMORY_USED = ["m_0015"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    train_dist = g.relative_mahalanobis_sq(tr.feats)
    eval_dist = g.relative_mahalanobis_sq(ev.feats)

    energy_cdf = ctx.h.normalizer("ecdf", train_energy)
    dist_cdf = ctx.h.normalizer("ecdf", train_dist)

    scores = energy_cdf(eval_energy) + dist_cdf(eval_dist)
    return np.asarray(scores, dtype=float)
