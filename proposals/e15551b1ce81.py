import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Bottleneck-space density is sensitive to near-OOS intent shifts, while pooled encoder "
    "density captures broader domain shifts. Train-fitted inverse-variance weights make their "
    "additive Mahalanobis fusion comparable without using evaluation statistics."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0003"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_bottleneck = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_encoder = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_b = g_bottleneck.min_mahalanobis_sq(tr.feats)
    train_e = g_encoder.min_mahalanobis_sq(tr.pooled)
    eval_b = g_bottleneck.min_mahalanobis_sq(ev.feats)
    eval_e = g_encoder.min_mahalanobis_sq(ev.pooled)

    mean_b = float(np.mean(train_b))
    mean_e = float(np.mean(train_e))
    var_b = float(np.var(train_b))
    var_e = float(np.var(train_e))

    eps = 1e-8
    weight_b = 1.0 / max(var_b, eps)
    weight_e = 1.0 / max(var_e, eps)
    total = weight_b + weight_e

    score_b = (eval_b - mean_b) / np.sqrt(max(var_b, eps))
    score_e = (eval_e - mean_e) / np.sqrt(max(var_e, eps))

    scores = (weight_b * score_b + weight_e * score_e) / total
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
