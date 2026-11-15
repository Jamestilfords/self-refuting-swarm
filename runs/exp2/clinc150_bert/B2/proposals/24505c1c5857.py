import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck and encoder-space Mahalanobis distances capture complementary near- and far-OOS deviations. Their additive weights are fitted from train-only robust distance scales so neither representation dominates merely because of its numerical range."
SOURCE = "https://github.com/kimjeyoung/Mahalanobis-BERT"
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0001"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    port_train = g_port.min_mahalanobis_sq(tr.feats)
    full_train = g_full.min_mahalanobis_sq(tr.pooled)
    port_eval = g_port.min_mahalanobis_sq(ev.feats)
    full_eval = g_full.min_mahalanobis_sq(ev.pooled)

    def robust_center_scale(x):
        center = np.median(x)
        lo, hi = np.percentile(x, [25.0, 75.0])
        scale = max(float(hi - lo), float(np.std(x)), 1e-6)
        return center, scale

    port_center, port_scale = robust_center_scale(port_train)
    full_center, full_scale = robust_center_scale(full_train)

    w_port = 1.0 / port_scale
    w_full = 1.0 / full_scale
    w_sum = w_port + w_full

    return (w_port * (port_eval - port_center) +
            w_full * (full_eval - full_center)) / w_sum
