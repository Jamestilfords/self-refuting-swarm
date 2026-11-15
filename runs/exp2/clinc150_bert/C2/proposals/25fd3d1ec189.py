import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy remains the dominant score for confident predictions, while an entropy-derived gate increasingly adds encoder-space Mahalanobis distance when the classifier is uncertain. This targets OOS examples whose logits are ambiguous even when their raw energy is not extreme."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_d = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_d = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    train_e = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_e = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    train_u = np.asarray(ctx.h.entropy(tr.logits), dtype=float)
    eval_u = np.asarray(ctx.h.entropy(ev.logits), dtype=float)

    d_med = np.median(train_d)
    d_iqr = np.percentile(train_d, 75) - np.percentile(train_d, 25)
    d_iqr = max(float(d_iqr), 1e-6)

    e_iqr = np.percentile(train_e, 75) - np.percentile(train_e, 25)
    e_iqr = max(float(e_iqr), 1e-6)

    u_center = np.percentile(train_u, 70)
    u_iqr = np.percentile(train_u, 75) - np.percentile(train_u, 25)
    u_iqr = max(float(u_iqr), 1e-6)

    gate = 1.0 / (1.0 + np.exp(-np.clip((eval_u - u_center) / u_iqr, -30.0, 30.0)))
    distance_residual = np.clip((eval_d - d_med) / d_iqr, -20.0, 20.0)

    scores = eval_e + gate * e_iqr * distance_residual
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
