import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy is retained for queries that look confidently in-scope, while a train-calibrated gate increasingly adds full encoder-space Mahalanobis distance when energy indicates uncertainty."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_mid = np.percentile(train_energy, 70.0)
    e_scale = np.percentile(train_energy, 90.0) - np.percentile(train_energy, 10.0)
    e_scale = max(float(e_scale), 1e-6)

    d_mid = np.median(train_dist)
    d_scale = np.percentile(train_dist, 90.0) - np.percentile(train_dist, 10.0)
    d_scale = max(float(d_scale), 1e-6)

    gate = 1.0 / (1.0 + np.exp(-np.clip((eval_energy - e_mid) / e_scale, -30.0, 30.0)))
    distance_term = (eval_dist - d_mid) / d_scale
    scores = eval_energy + gate * distance_term

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
