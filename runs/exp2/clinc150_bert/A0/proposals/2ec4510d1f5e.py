import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy remains the primary score for confident predictions, while a train-calibrated Mahalanobis residual is activated mainly in the uncertain energy tail. This emphasizes feature-space novelty when classifier uncertainty alone is ambiguous."
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

    e_center = np.quantile(train_energy, 0.75)
    e_scale = np.quantile(train_energy, 0.90) - np.quantile(train_energy, 0.50)
    e_scale = max(float(e_scale), 1e-6)

    d_center = np.median(train_dist)
    d_scale = np.quantile(train_dist, 0.75) - np.quantile(train_dist, 0.25)
    d_scale = max(float(d_scale), 1e-6)

    gate = 1.0 / (1.0 + np.exp(-np.clip((eval_energy - e_center) / e_scale, -40.0, 40.0)))
    distance_residual = (eval_dist - d_center) / d_scale
    scores = eval_energy + gate * distance_residual

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
