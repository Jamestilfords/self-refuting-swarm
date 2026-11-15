import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy remains the primary score for confident predictions, while a train-calibrated uncertainty gate increasingly adds pooled-feature Mahalanobis distance when the logit energy indicates uncertainty. This targets OOS examples whose classifier confidence alone is ambiguous."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    gda = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_dist = gda.min_mahalanobis_sq(tr.pooled)
    eval_dist = gda.min_mahalanobis_sq(ev.pooled)

    e_med = np.median(train_energy)
    e_scale = np.percentile(train_energy, 75) - np.percentile(train_energy, 25)
    e_scale = max(float(e_scale), 1e-6)

    d_med = np.median(train_dist)
    d_scale = np.percentile(train_dist, 75) - np.percentile(train_dist, 25)
    d_scale = max(float(d_scale), 1e-6)

    energy_z = (eval_energy - e_med) / e_scale
    dist_z = (eval_dist - d_med) / d_scale

    gate = 1.0 / (1.0 + np.exp(-np.clip(energy_z - 0.5, -30.0, 30.0)))
    return energy_z + gate * dist_z
