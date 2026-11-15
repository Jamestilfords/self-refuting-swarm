import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck and encoder-space Gaussian distances capture complementary near- and far-OOS deviations. Training-set robust scaling makes their additive fusion comparable without using evaluation-set statistics."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_feat = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    g_pool = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_feat = np.asarray(g_feat.min_mahalanobis_sq(tr.feats), dtype=float)
    train_pool = np.asarray(g_pool.min_mahalanobis_sq(tr.pooled), dtype=float)

    def robust_scale(x):
        q25, q75 = np.quantile(x, [0.25, 0.75])
        scale = q75 - q25
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = np.std(x)
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = 1.0
        return float(np.median(x)), float(scale)

    med_feat, scale_feat = robust_scale(train_feat)
    med_pool, scale_pool = robust_scale(train_pool)

    # Equal contributions after train-only robust calibration.
    w_feat = 0.5 / scale_feat
    w_pool = 0.5 / scale_pool

    eval_feat = np.asarray(g_feat.min_mahalanobis_sq(ev.feats), dtype=float)
    eval_pool = np.asarray(g_pool.min_mahalanobis_sq(ev.pooled), dtype=float)

    scores = w_feat * (eval_feat - med_feat) + w_pool * (eval_pool - med_pool)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
