import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Mahalanobis distance in bottleneck space is sensitive to nearby held-out intents, while encoder-space distance captures broader semantic domain shifts. Training-set robust scales fit the additive fusion weights without using evaluation statistics."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    port_train = g_port.min_mahalanobis_sq(tr.feats)
    full_train = g_full.min_mahalanobis_sq(tr.pooled)

    port_med = np.median(port_train)
    full_med = np.median(full_train)

    port_scale = np.median(np.abs(port_train - port_med))
    full_scale = np.median(np.abs(full_train - full_med))

    if not np.isfinite(port_scale) or port_scale < 1e-8:
        port_scale = np.std(port_train) + 1e-8
    if not np.isfinite(full_scale) or full_scale < 1e-8:
        full_scale = np.std(full_train) + 1e-8

    w_port = 1.0 / port_scale
    w_full = 1.0 / full_scale
    weight_sum = w_port + w_full

    port_eval = g_port.min_mahalanobis_sq(ev.feats)
    full_eval = g_full.min_mahalanobis_sq(ev.pooled)

    scores = (
        w_port * (port_eval - port_med)
        + w_full * (full_eval - full_med)
    ) / weight_sum

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
