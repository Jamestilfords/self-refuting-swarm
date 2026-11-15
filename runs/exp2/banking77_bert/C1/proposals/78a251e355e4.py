import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck Mahalanobis distance is sensitive to nearby held-out intents, while encoder-space distance captures broader domain shifts. Training-set robust scaling and inverse-variation weights put both density signals on a comparable additive scale without using evaluation statistics."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    port_train = g_port.min_mahalanobis_sq(tr.feats)
    full_train = g_full.min_mahalanobis_sq(tr.pooled)
    port_eval = g_port.min_mahalanobis_sq(ev.feats)
    full_eval = g_full.min_mahalanobis_sq(ev.pooled)

    def robust_scale(train_values, eval_values):
        center = np.median(train_values)
        scale = np.median(np.abs(train_values - center)) * 1.4826
        if not np.isfinite(scale) or scale < 1e-8:
            scale = np.std(train_values)
        scale = max(float(scale), 1e-8)
        return (eval_values - center) / scale, scale

    z_port, s_port = robust_scale(port_train, port_eval)
    z_full, s_full = robust_scale(full_train, full_eval)

    w_port = 1.0 / s_port
    w_full = 1.0 / s_full
    total = w_port + w_full
    w_port /= total
    w_full /= total

    return w_port * z_port + w_full * z_full
