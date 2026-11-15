import numpy as np
from scipy.special import ndtri

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Empirical train-distribution calibration converts energy and pooled-feature distance into comparable Gaussian quantile scales. A soft maximum emphasizes queries that are anomalous in either logits or representation space while changing the ranking from a linear blend."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    def gaussian_ecdf(train_values, values):
        reference = np.sort(np.asarray(train_values, dtype=float))
        n = reference.size
        ranks = np.searchsorted(reference, values, side="right")
        probabilities = (ranks + 0.5) / (n + 1.0)
        probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
        return ndtri(probabilities)

    energy_scale = gaussian_ecdf(train_energy, eval_energy)
    dist_scale = gaussian_ecdf(train_dist, eval_dist)

    return np.logaddexp(energy_scale, dist_scale)
