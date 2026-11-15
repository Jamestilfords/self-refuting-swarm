import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck and encoder-space class-conditional distances capture complementary near-OOS and far-OOS deviations. Each density is robustly calibrated and weighted using only its training-set dispersion before additive fusion."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    d_port_train = g_port.min_mahalanobis_sq(tr.feats)
    d_full_train = g_full.min_mahalanobis_sq(tr.pooled)

    def robust_scale(x):
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        return med, max(1.4826 * mad, 1e-6)

    port_med, port_scale = robust_scale(d_port_train)
    full_med, full_scale = robust_scale(d_full_train)

    z_port_train = (d_port_train - port_med) / port_scale
    z_full_train = (d_full_train - full_med) / full_scale

    port_var = max(np.var(z_port_train), 1e-6)
    full_var = max(np.var(z_full_train), 1e-6)
    w_port = 1.0 / port_var
    w_full = 1.0 / full_var
    weight_sum = w_port + w_full
    w_port /= weight_sum
    w_full /= weight_sum

    d_port_eval = g_port.min_mahalanobis_sq(ev.feats)
    d_full_eval = g_full.min_mahalanobis_sq(ev.pooled)

    z_port_eval = (d_port_eval - port_med) / port_scale
    z_full_eval = (d_full_eval - full_med) / full_scale

    return w_port * z_port_eval + w_full * z_full_eval
