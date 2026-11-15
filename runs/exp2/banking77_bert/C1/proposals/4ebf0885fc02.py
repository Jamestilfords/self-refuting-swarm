import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy captures classifier uncertainty while tied-covariance feature distance captures representation-space novelty. Empirical train CDF calibration puts both signals on a common scale without using evaluation-set statistics."
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

    energy_ref = np.sort(np.asarray(train_energy, dtype=float))
    dist_ref = np.sort(np.asarray(train_dist, dtype=float))

    energy_u = np.searchsorted(energy_ref, eval_energy, side="right") / float(len(energy_ref))
    dist_u = np.searchsorted(dist_ref, eval_dist, side="right") / float(len(dist_ref))

    return 0.5 * energy_u + 0.5 * dist_u
