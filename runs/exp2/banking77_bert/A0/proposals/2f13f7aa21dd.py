import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Empirical CDF calibration puts logit energy and encoder-space Mahalanobis distance on the same train-derived scale. Their geometric fusion emphasizes queries that are simultaneously classifier-uncertain and atypical in representation space."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0001"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    e_sorted = np.sort(np.nan_to_num(
        train_energy, nan=0.0, posinf=1e12, neginf=-1e12
    ))
    d_sorted = np.sort(np.nan_to_num(
        train_dist, nan=0.0, posinf=1e12, neginf=-1e12
    ))

    eval_energy = np.nan_to_num(eval_energy, nan=0.0, posinf=1e12, neginf=-1e12)
    eval_dist = np.nan_to_num(eval_dist, nan=0.0, posinf=1e12, neginf=-1e12)

    n = float(len(tr.labels) + 1)
    energy_tail = np.searchsorted(e_sorted, eval_energy, side="right") / n
    dist_tail = np.searchsorted(d_sorted, eval_dist, side="right") / n

    return np.sqrt(energy_tail * dist_tail)
