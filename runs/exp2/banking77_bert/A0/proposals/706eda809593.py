import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck Mahalanobis distance is sensitive to nearby held-out intents, while encoder-space distance captures broader domain shifts. Train-fitted robust scaling makes the two density signals comparable before additive fusion."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    port_train = g_port.min_mahalanobis_sq(tr.feats)
    full_train = g_full.min_mahalanobis_sq(tr.pooled)
    port_eval = g_port.min_mahalanobis_sq(ev.feats)
    full_eval = g_full.min_mahalanobis_sq(ev.pooled)

    def robust_location_scale(x):
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = max(1.4826 * float(mad), 1e-6)
        return float(med), scale

    port_med, port_scale = robust_location_scale(port_train)
    full_med, full_scale = robust_location_scale(full_train)

    port_weight = 1.0 / port_scale
    full_weight = 1.0 / full_scale
    weight_sum = port_weight + full_weight

    fused = (
        port_weight * (port_eval - port_med)
        + full_weight * (full_eval - full_med)
    ) / max(weight_sum, 1e-12)

    return np.nan_to_num(fused, nan=0.0, posinf=1e12, neginf=-1e12)
