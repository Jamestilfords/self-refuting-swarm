import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "This preserves the energy ranking for training-like confident predictions, while smoothly activating pooled-feature Mahalanobis distance when energy indicates uncertainty. Feature distance can then distinguish uncertain in-scope boundary cases from genuinely unfamiliar queries."
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

    e_center = np.median(train_energy)
    e_scale = np.quantile(train_energy, 0.75) - np.quantile(train_energy, 0.25)
    d_center = np.median(train_dist)
    d_scale = np.quantile(train_dist, 0.75) - np.quantile(train_dist, 0.25)

    e_scale = max(float(e_scale), 1e-6)
    d_scale = max(float(d_scale), 1e-6)

    e_z = (eval_energy - e_center) / e_scale
    d_z = (eval_dist - d_center) / d_scale

    gate = 1.0 / (1.0 + np.exp(-np.clip(e_z - 1.0, -40.0, 40.0)))
    scores = e_z + gate * d_z
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
