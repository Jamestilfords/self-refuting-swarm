import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy retains its ordering for train-calibrated confident queries, while a smooth "
    "uncertainty gate progressively replaces it with pooled-feature support distance."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0065", "m_0069"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    e_sorted = np.sort(train_energy)
    d_sorted = np.sort(train_dist)
    n = float(len(tr.labels))

    e_rank = np.searchsorted(e_sorted, eval_energy, side="right") / n
    d_rank = np.searchsorted(d_sorted, eval_dist, side="right") / n

    start = np.quantile(train_energy, 0.65)
    end = np.quantile(train_energy, 0.90)
    width = max(float(end - start), 1e-8)
    gate = np.clip((eval_energy - start) / width, 0.0, 1.0)

    return (1.0 - gate) * e_rank + gate * d_rank
