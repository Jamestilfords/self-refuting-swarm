import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck Mahalanobis distance is sensitive to nearby held-out intents, while encoder-space distance captures broader domain shifts. Train-fitted robust scaling makes their additive combination balance both density signals without using evaluation statistics."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_port = g_port.min_mahalanobis_sq(tr.feats)
    train_full = g_full.min_mahalanobis_sq(tr.pooled)
    eval_port = g_port.min_mahalanobis_sq(ev.feats)
    eval_full = g_full.min_mahalanobis_sq(ev.pooled)

    def robust_scale(x):
        q10, q50, q90 = np.quantile(x, [0.10, 0.50, 0.90])
        scale = q90 - q10
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        return q50, scale

    med_port, scale_port = robust_scale(train_port)
    med_full, scale_full = robust_scale(train_full)

    z_port = (eval_port - med_port) / scale_port
    z_full = (eval_full - med_full) / scale_full
    return np.asarray(z_port + z_full, dtype=float)
