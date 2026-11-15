import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck Mahalanobis distance is sensitive to near-OOS queries that depart from learned intent representations, while encoder-space distance captures broader domain shifts. Train-fitted robust scaling gives both density scores comparable additive influence without using evaluation-set statistics."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    port_train = g_port.min_mahalanobis_sq(tr.feats)
    full_train = g_full.min_mahalanobis_sq(tr.pooled)

    port_med = np.median(port_train)
    full_med = np.median(full_train)
    port_scale = np.percentile(port_train, 90) - np.percentile(port_train, 50)
    full_scale = np.percentile(full_train, 90) - np.percentile(full_train, 50)

    port_scale = max(float(port_scale), 1e-8)
    full_scale = max(float(full_scale), 1e-8)

    inv_port = 1.0 / port_scale
    inv_full = 1.0 / full_scale
    total = inv_port + inv_full
    w_port = inv_port / total
    w_full = inv_full / total

    port_eval = g_port.min_mahalanobis_sq(ev.feats)
    full_eval = g_full.min_mahalanobis_sq(ev.pooled)

    port_score = (port_eval - port_med) / port_scale
    full_score = (full_eval - full_med) / full_scale
    scores = w_port * port_score + w_full * full_score

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
