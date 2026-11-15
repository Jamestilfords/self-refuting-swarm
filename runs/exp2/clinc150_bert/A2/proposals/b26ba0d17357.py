import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck Mahalanobis distance is sensitive to nearby held-out intents, while encoder-space distance captures broader domain shifts. Training-set dispersion calibrates their additive weights without using evaluation statistics."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_port = np.asarray(g_port.min_mahalanobis_sq(tr.feats), dtype=float)
    train_full = np.asarray(g_full.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_port = np.asarray(g_port.min_mahalanobis_sq(ev.feats), dtype=float)
    eval_full = np.asarray(g_full.min_mahalanobis_sq(ev.pooled), dtype=float)

    scale_port = max(float(np.std(train_port)), 1e-8)
    scale_full = max(float(np.std(train_full)), 1e-8)

    weights = np.array([1.0 / scale_port, 1.0 / scale_full])
    weights /= np.sum(weights)

    scores = weights[0] * eval_port + weights[1] * eval_full
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
