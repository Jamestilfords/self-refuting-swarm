import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Bottleneck Mahalanobis distance is sensitive to nearby unseen intents, while encoder-space "
    "Mahalanobis distance better captures broader semantic and domain shifts. Robust train-fitted "
    "median/IQR calibration makes their additive combination insensitive to differing distance scales."
)
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0003"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    port_train = g_port.min_mahalanobis_sq(tr.feats)
    full_train = g_full.min_mahalanobis_sq(tr.pooled)
    port_eval = g_port.min_mahalanobis_sq(ev.feats)
    full_eval = g_full.min_mahalanobis_sq(ev.pooled)

    def robust_standardize(train_values, values):
        center = np.median(train_values)
        q1, q3 = np.percentile(train_values, [25.0, 75.0])
        scale = q3 - q1
        if not np.isfinite(scale) or scale < 1e-8:
            scale = np.std(train_values)
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        return (values - center) / scale

    port_z = robust_standardize(port_train, port_eval)
    full_z = robust_standardize(full_train, full_eval)
    return 0.5 * port_z + 0.5 * full_z
