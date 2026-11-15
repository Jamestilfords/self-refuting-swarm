import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and pooled-feature Mahalanobis distance capture complementary logit uncertainty and representation atypicality. Empirical train-only CDF calibration puts both signals on a common in-distribution tail scale before combining them."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0117"]


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

    energy_tail = np.searchsorted(energy_ref, eval_energy, side="right") / n
    dist_tail = np.searchsorted(dist_ref, eval_dist, side="right") / n

    return 0.5 * energy_tail + 0.5 * dist_tail
