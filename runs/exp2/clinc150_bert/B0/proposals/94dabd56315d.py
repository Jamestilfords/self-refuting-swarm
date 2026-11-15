import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy remains the primary signal for classifier-confident queries, while a train-calibrated encoder Mahalanobis distance is increasingly weighted as energy indicates uncertainty. This targets queries whose uncertain logits require feature-manifold evidence for separation."
SOURCE = "https://www.emergentmind.com/papers/2101.03778"
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0001", "m_0017"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    energy_ref = np.sort(train_energy)
    dist_ref = np.sort(train_dist)
    n_energy = float(max(1, energy_ref.size))
    n_dist = float(max(1, dist_ref.size))

    energy_q = np.searchsorted(energy_ref, eval_energy, side="right") / n_energy
    dist_q = np.searchsorted(dist_ref, eval_dist, side="right") / n_dist

    gate = energy_q * energy_q
    return energy_q + gate * dist_q
