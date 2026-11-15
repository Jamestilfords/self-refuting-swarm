import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and shared-covariance feature distance capture complementary logit-space and representation-space uncertainty. Empirical CDF calibration on training scores places both signals on a common, robust scale without using evaluation-set statistics."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    train_energy = np.nan_to_num(train_energy, nan=0.0, posinf=1e12, neginf=-1e12)
    eval_energy = np.nan_to_num(eval_energy, nan=0.0, posinf=1e12, neginf=-1e12)
    train_dist = np.nan_to_num(train_dist, nan=0.0, posinf=1e12, neginf=-1e12)
    eval_dist = np.nan_to_num(eval_dist, nan=0.0, posinf=1e12, neginf=-1e12)

    energy_ref = np.sort(train_energy)
    dist_ref = np.sort(train_dist)
    n = float(len(tr.labels))

    energy_q = np.searchsorted(energy_ref, eval_energy, side="right") / n
    dist_q = np.searchsorted(dist_ref, eval_dist, side="right") / n

    return 0.5 * energy_q + 0.5 * dist_q
