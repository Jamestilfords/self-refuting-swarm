import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck Gaussian distance is sensitive to near-OOS intent shifts, while encoder-output distance captures broader domain shifts. Robust train-fitted scaling puts the two density signals on comparable additive scales without using evaluation statistics."
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

    def center_scale(values):
        center = np.median(values)
        q25, q75 = np.quantile(values, [0.25, 0.75])
        scale = q75 - q25
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = np.std(values)
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        return center, scale

    port_center, port_scale = center_scale(port_train)
    full_center, full_scale = center_scale(full_train)

    scores = (port_eval - port_center) / port_scale
    scores += (full_eval - full_center) / full_scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
