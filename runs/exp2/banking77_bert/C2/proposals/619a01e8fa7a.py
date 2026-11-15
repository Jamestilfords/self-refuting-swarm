import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "This keeps the energy score dominant for items with in-distribution-like confidence, while smoothly shifting weight toward pooled-feature Mahalanobis distance when energy indicates uncertainty. Train-set robust calibration makes the two signals comparable without using evaluation-set statistics."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    e_med = np.median(train_energy)
    d_med = np.median(train_dist)
    e_scale = np.median(np.abs(train_energy - e_med)) * 1.4826
    d_scale = np.median(np.abs(train_dist - d_med)) * 1.4826
    e_scale = max(float(e_scale), 1e-6)
    d_scale = max(float(d_scale), 1e-6)

    e_z = np.clip((eval_energy - e_med) / e_scale, -30.0, 30.0)
    d_z = np.clip((eval_dist - d_med) / d_scale, -30.0, 30.0)

    uncertainty_center = np.quantile(train_energy, 0.75)
    gate = 1.0 / (1.0 + np.exp(-np.clip(
        (eval_energy - uncertainty_center) / e_scale, -30.0, 30.0
    )))

    return (1.0 - gate) * e_z + gate * d_z
