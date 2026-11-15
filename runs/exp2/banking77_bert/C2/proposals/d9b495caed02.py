import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck Mahalanobis distance is sensitive to nearby held-out intents, while encoder-space distance captures broader domain shift. Each density is robustly scaled using only its in-scope training distribution before additive fusion."
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

    def robust_scale(x):
        q25, q75 = np.percentile(x, [25.0, 75.0])
        scale = q75 - q25
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = np.std(x)
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = 1.0
        return scale

    port_scale = robust_scale(train_port)
    full_scale = robust_scale(train_full)

    score_port = eval_port / port_scale
    score_full = eval_full / full_scale
    return np.nan_to_num(score_port + score_full, nan=0.0, posinf=1e12, neginf=-1e12)
