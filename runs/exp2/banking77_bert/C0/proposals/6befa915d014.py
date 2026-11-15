import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Train-derived empirical tail scales place energy and Mahalanobis displacement on a shared percentile scale. A soft-OR fusion prioritizes queries that are extreme on either classifier uncertainty or feature-space distance, while producing a ranking distinct from linear addition."
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
    n_e = max(e_sorted.size, 1)
    n_d = max(d_sorted.size, 1)

    e_pct = np.searchsorted(e_sorted, eval_energy, side="right") / n_e
    d_pct = np.searchsorted(d_sorted, eval_dist, side="right") / n_d

    e_pct = np.clip(e_pct, 0.0, 1.0)
    d_pct = np.clip(d_pct, 0.0, 1.0)
    return e_pct + d_pct - e_pct * d_pct
