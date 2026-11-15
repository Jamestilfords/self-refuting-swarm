import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy captures classifier-level uncertainty while encoder Mahalanobis distance captures representation novelty. Calibrating both against their respective training distributions puts their tail evidence on a common scale before fusion."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0025", "m_0043"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.nan_to_num(
        ctx.h.energy(tr.logits), nan=0.0, posinf=1e12, neginf=-1e12
    )
    eval_energy = np.nan_to_num(
        ctx.h.energy(ev.logits), nan=0.0, posinf=1e12, neginf=-1e12
    )

    train_dist = np.nan_to_num(
        g.min_mahalanobis_sq(tr.pooled), nan=0.0, posinf=1e12, neginf=-1e12
    )
    eval_dist = np.nan_to_num(
        g.min_mahalanobis_sq(ev.pooled), nan=0.0, posinf=1e12, neginf=-1e12
    )

    energy_sorted = np.sort(train_energy)
    dist_sorted = np.sort(train_dist)
    n = float(len(tr.labels))

    energy_tail = np.searchsorted(energy_sorted, eval_energy, side="right") / n
    dist_tail = np.searchsorted(dist_sorted, eval_dist, side="right") / n

    return 0.5 * energy_tail + 0.5 * dist_tail
