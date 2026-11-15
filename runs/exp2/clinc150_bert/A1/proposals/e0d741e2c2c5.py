import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck Mahalanobis distance is sensitive to near-OOS intent shifts, while pooled encoder distance captures broader domain shifts. Training-set robust scaling makes their additive fusion comparable without using evaluation-set statistics."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_port = g_port.min_mahalanobis_sq(tr.feats)
    train_full = g_full.min_mahalanobis_sq(tr.pooled)

    port_med = np.median(train_port)
    full_med = np.median(train_full)
    port_iqr = np.percentile(train_port, 75) - np.percentile(train_port, 25)
    full_iqr = np.percentile(train_full, 75) - np.percentile(train_full, 25)

    port_scale = max(float(port_iqr), 1e-6)
    full_scale = max(float(full_iqr), 1e-6)

    # Inverse robust dispersion gives each density a train-fitted comparable weight.
    w_port = 1.0 / port_scale
    w_full = 1.0 / full_scale
    weight_sum = w_port + w_full
    w_port /= weight_sum
    w_full /= weight_sum

    d_port = g_port.min_mahalanobis_sq(ev.feats)
    d_full = g_full.min_mahalanobis_sq(ev.pooled)

    z_port = (d_port - port_med) / port_scale
    z_full = (d_full - full_med) / full_scale
    return np.asarray(w_port * z_port + w_full * z_full, dtype=float)
