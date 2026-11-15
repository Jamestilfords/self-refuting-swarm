import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy remains the dominant score for confidently classified inputs, while a train-calibrated pooled-feature Mahalanobis distance is increasingly activated only when logits indicate uncertainty."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0039"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    e_sorted = np.sort(train_energy)
    d_sorted = np.sort(train_dist)
    n = max(len(train_energy), 1)

    e_train_pct = np.searchsorted(e_sorted, train_energy, side="right") / n
    e_eval_pct = np.searchsorted(e_sorted, eval_energy, side="right") / n
    d_eval_pct = np.searchsorted(d_sorted, eval_dist, side="right") / n

    center = float(np.median(e_train_pct))
    scale = float(np.median(np.abs(e_train_pct - center)))
    scale = max(scale, 1e-6)

    uncertainty_gate = 1.0 / (1.0 + np.exp(-(e_eval_pct - center) / scale))
    return e_eval_pct + uncertainty_gate * d_eval_pct
