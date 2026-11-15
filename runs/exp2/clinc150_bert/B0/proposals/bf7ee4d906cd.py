import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck and encoder-space Mahalanobis densities capture complementary near-intent and far-domain deviations. Train-distribution scaling makes their additive contributions comparable without using evaluation-set statistics."
SOURCE = "https://www.emergentmind.com/papers/2101.03778"
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0001"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    port_train = g_port.min_mahalanobis_sq(tr.feats)
    full_train = g_full.min_mahalanobis_sq(tr.pooled)
    port_eval = g_port.min_mahalanobis_sq(ev.feats)
    full_eval = g_full.min_mahalanobis_sq(ev.pooled)

    port_center = np.median(port_train)
    full_center = np.median(full_train)
    port_scale = np.median(np.abs(port_train - port_center)) * 1.4826
    full_scale = np.median(np.abs(full_train - full_center)) * 1.4826

    port_scale = max(float(port_scale), 1e-6)
    full_scale = max(float(full_scale), 1e-6)

    port_weight = 1.0 / port_scale
    full_weight = 1.0 / full_scale
    weight_sum = port_weight + full_weight
    port_weight /= weight_sum
    full_weight /= weight_sum

    scores = (
        port_weight * (port_eval - port_center) / port_scale
        + full_weight * (full_eval - full_center) / full_scale
    )
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
