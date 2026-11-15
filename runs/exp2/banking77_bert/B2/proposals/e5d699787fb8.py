import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck and encoder-space Mahalanobis distances capture complementary near-OOS and far-OOS deviations. Each density is robustly scaled using only its in-scope training distribution before additive fusion."
SOURCE = "https://www.emergentmind.com/papers/2101.03778"
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0001"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_port = g_port.min_mahalanobis_sq(tr.feats)
    train_full = g_full.min_mahalanobis_sq(tr.pooled)
    eval_port = g_port.min_mahalanobis_sq(ev.feats)
    eval_full = g_full.min_mahalanobis_sq(ev.pooled)

    p10_port, p90_port = np.percentile(train_port, [10.0, 90.0])
    p10_full, p90_full = np.percentile(train_full, [10.0, 90.0])

    scale_port = max(float(p90_port - p10_port), 1e-8)
    scale_full = max(float(p90_full - p10_full), 1e-8)

    fused = (eval_port - p10_port) / scale_port
    fused += (eval_full - p10_full) / scale_full
    return np.nan_to_num(fused, nan=0.0, posinf=1e12, neginf=-1e12)
