import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Empirical-CDF calibration puts classifier energy and pooled-feature Mahalanobis novelty on a shared train-derived scale. Summing their upper-tail evidence detects queries that are uncertain in logit space, atypical in representation space, or both."
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

    energy_sorted = np.sort(train_energy)
    dist_sorted = np.sort(train_dist)
    n = float(len(tr.labels))

    energy_q = (np.searchsorted(energy_sorted, eval_energy, side="right") + 0.5) / (n + 1.0)
    dist_q = (np.searchsorted(dist_sorted, eval_dist, side="right") + 0.5) / (n + 1.0)

    energy_tail = -np.log1p(-energy_q)
    dist_tail = -np.log1p(-dist_q)
    return 0.5 * (energy_tail + dist_tail)
