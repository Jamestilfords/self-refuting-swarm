import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck Mahalanobis distance is sensitive to near-OOS departures among related intents, while pooled encoder distance captures broader domain shifts. Train-fitted robust scales make their additive combination comparable without using evaluation-set statistics."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0004"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    port_train = g_port.min_mahalanobis_sq(tr.feats)
    full_train = g_full.min_mahalanobis_sq(tr.pooled)

    port_med = np.median(port_train)
    full_med = np.median(full_train)
    port_scale = np.quantile(port_train, 0.75) - np.quantile(port_train, 0.25)
    full_scale = np.quantile(full_train, 0.75) - np.quantile(full_train, 0.25)

    port_scale = max(float(port_scale), 1e-8)
    full_scale = max(float(full_scale), 1e-8)

    w_port = 1.0 / port_scale
    w_full = 1.0 / full_scale
    weight_sum = w_port + w_full
    w_port /= weight_sum
    w_full /= weight_sum

    port_eval = (g_port.min_mahalanobis_sq(ev.feats) - port_med) / port_scale
    full_eval = (g_full.min_mahalanobis_sq(ev.pooled) - full_med) / full_scale

    return w_port * port_eval + w_full * full_eval
