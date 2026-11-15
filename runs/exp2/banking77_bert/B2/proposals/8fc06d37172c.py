import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Confident in-scope predictions retain the energy ranking, while samples in the "
    "train-calibrated uncertain-energy tail receive an additional pooled-feature support penalty."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0054", "m_0058"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    threshold = np.percentile(train_energy, 60.0)
    upper = np.percentile(train_energy, 85.0)
    width = max(upper - threshold, 1e-6)
    gate = np.clip((eval_energy - threshold) / width, 0.0, 1.0)

    dist_center = np.median(train_dist)
    dist_scale = np.percentile(train_dist, 75.0) - np.percentile(train_dist, 25.0)
    dist_scale = max(dist_scale, 1e-6)

    energy_scale = np.percentile(train_energy, 75.0) - np.percentile(train_energy, 25.0)
    energy_scale = max(energy_scale, 1e-6)

    distance_term = energy_scale * (eval_dist - dist_center) / dist_scale
    scores = eval_energy + gate * distance_term
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
