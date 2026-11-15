import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and pooled-space Mahalanobis distance capture classifier uncertainty and representation atypicality on different raw scales. Train-fitted empirical percentiles put both signals on a common tail-probability scale before averaging them."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0059", "m_0065"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_d = g.min_mahalanobis_sq(tr.pooled)
    eval_d = g.min_mahalanobis_sq(ev.pooled)

    train_e = ctx.h.energy(tr.logits)
    eval_e = ctx.h.energy(ev.logits)

    d_sorted = np.sort(np.asarray(train_d, dtype=float))
    e_sorted = np.sort(np.asarray(train_e, dtype=float))
    n_d = max(len(d_sorted), 1)
    n_e = max(len(e_sorted), 1)

    d_pct = np.searchsorted(d_sorted, eval_d, side="right") / n_d
    e_pct = np.searchsorted(e_sorted, eval_e, side="right") / n_e

    return 0.5 * d_pct + 0.5 * e_pct
