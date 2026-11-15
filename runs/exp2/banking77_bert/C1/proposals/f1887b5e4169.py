import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Standardizing logit energy and full encoder-space Mahalanobis distance using only in-scope training statistics places complementary confidence signals on a common scale. Queries that are both low-confidence and atypical in representation space receive high OOS scores."
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

    e_mean = np.mean(train_energy)
    d_mean = np.mean(train_dist)
    e_scale = np.std(train_energy) + 1e-8
    d_scale = np.std(train_dist) + 1e-8

    energy_z = (eval_energy - e_mean) / e_scale
    dist_z = (eval_dist - d_mean) / d_scale
    return np.nan_to_num(0.5 * (energy_z + dist_z), nan=0.0, posinf=1e6, neginf=-1e6)
