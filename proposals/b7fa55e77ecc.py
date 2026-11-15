import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and pooled-space Mahalanobis distance capture complementary classifier uncertainty and representation support. Mapping each score through its train-fitted empirical tail probability makes their scales comparable without using evaluation-set statistics."
SOURCE = "https://github.com/kimjeyoung/Mahalanobis-BERT"
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0001", "m_0010"]


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

    e_tail = np.searchsorted(energy_ref, eval_energy, side="right") / n
    d_tail = np.searchsorted(dist_ref, eval_dist, side="right") / n

    eps = 1.0 / (n + 2.0)
    e_tail = np.clip(e_tail, eps, 1.0 - eps)
    d_tail = np.clip(d_tail, eps, 1.0 - eps)

    return np.log(e_tail / (1.0 - e_tail)) + np.log(d_tail / (1.0 - d_tail))
