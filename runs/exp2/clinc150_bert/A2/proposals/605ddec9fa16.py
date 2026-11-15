import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy is retained as a monotone confidence score for training-supported, confident predictions, "
    "while a train-calibrated gate routes energy-uncertain queries toward pooled-space class distance."
)
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

    e_sorted = np.sort(train_energy)
    d_sorted = np.sort(train_dist)
    n = max(len(tr.labels), 1)

    e_tail = np.searchsorted(e_sorted, eval_energy, side="right") / n
    d_tail = np.searchsorted(d_sorted, eval_dist, side="right") / n

    lo = np.quantile(train_energy, 0.60)
    hi = np.quantile(train_energy, 0.90)
    scale = max(float(hi - lo), 1e-8)
    gate = 1.0 / (1.0 + np.exp(-6.0 * (eval_energy - lo) / scale))

    return (1.0 - gate) * e_tail + gate * d_tail
