import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Bottleneck Mahalanobis distance is sensitive to nearby held-out intents, while encoder-space "
    "Mahalanobis distance captures broader domain shifts. Train-fitted robust scaling and inverse-spread "
    "weights make their additive fusion comparable without using evaluation-set statistics."
)
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_port = np.asarray(g_port.min_mahalanobis_sq(tr.feats), dtype=float)
    train_full = np.asarray(g_full.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_port = np.asarray(g_port.min_mahalanobis_sq(ev.feats), dtype=float)
    eval_full = np.asarray(g_full.min_mahalanobis_sq(ev.pooled), dtype=float)

    def robust_location_scale(x):
        med = np.median(x)
        q25, q75 = np.quantile(x, [0.25, 0.75])
        scale = max(float(q75 - q25), 1e-6)
        return med, scale

    med_port, scale_port = robust_location_scale(train_port)
    med_full, scale_full = robust_location_scale(train_full)

    raw_weights = np.array([1.0 / scale_port, 1.0 / scale_full])
    weights = raw_weights / np.sum(raw_weights)

    z_port = (eval_port - med_port) / scale_port
    z_full = (eval_full - med_full) / scale_full
    scores = weights[0] * z_port + weights[1] * z_full

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
