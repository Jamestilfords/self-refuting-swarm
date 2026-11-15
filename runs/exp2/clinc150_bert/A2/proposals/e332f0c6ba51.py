import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Confident items retain the energy ordering, while items with unusually high "
    "energy smoothly transition toward encoder-space Mahalanobis novelty. This "
    "lets feature distance dominate only in the uncertain regime where logits are less reliable."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0043"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_med = np.median(train_energy)
    e_iqr = np.quantile(train_energy, 0.75) - np.quantile(train_energy, 0.25)
    d_med = np.median(train_dist)
    d_iqr = np.quantile(train_dist, 0.75) - np.quantile(train_dist, 0.25)

    e_scale = max(float(e_iqr), 1e-6)
    d_scale = max(float(d_iqr), 1e-6)

    e_norm = (eval_energy - e_med) / e_scale
    d_norm = (eval_dist - d_med) / d_scale

    transition = np.quantile(train_energy, 0.75)
    sharpness = max(float(np.quantile(train_energy, 0.90) - transition), 1e-6)
    gate = 1.0 / (1.0 + np.exp(-np.clip((eval_energy - transition) / sharpness, -30.0, 30.0)))

    return (1.0 - gate) * e_norm + gate * d_norm
