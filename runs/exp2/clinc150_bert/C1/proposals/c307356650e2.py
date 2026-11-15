import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "This keeps the raw energy score for training-like confident predictions, while smoothly replacing it with a feature-space Mahalanobis distance once energy exceeds the in-scope uncertainty region. The distance is robustly rescaled using only training statistics so the transition remains calibrated without using evaluation-set statistics."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_lo, e_hi = np.quantile(train_energy, [0.80, 0.95])
    d_med = np.median(train_dist)
    d_mad = np.median(np.abs(train_dist - d_med))
    e_med = np.median(train_energy)
    e_mad = np.median(np.abs(train_energy - e_med))

    d_scale = max(1.4826 * d_mad, 1e-8)
    e_scale = max(1.4826 * e_mad, 1e-8)
    dist_on_energy_scale = e_med + e_scale * (eval_dist - d_med) / d_scale

    width = max(e_hi - e_lo, 1e-8)
    gate = np.clip((eval_energy - e_lo) / width, 0.0, 1.0)

    scores = (1.0 - gate) * eval_energy + gate * dist_on_energy_scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
