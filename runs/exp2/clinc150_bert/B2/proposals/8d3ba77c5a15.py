import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Combining local support in the compact bottleneck space with angular semantic support in pooled embeddings captures complementary forms of in-scope neighborhood structure. Train-only robust scaling prevents either feature geometry from dominating solely because of its distance scale."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0129"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    tr_feat = ctx.h.knn(tr.feats, tr.feats, k=15, l2=False)
    ev_feat = ctx.h.knn(tr.feats, ev.feats, k=15, l2=False)

    tr_pool = ctx.h.knn(tr.pooled, tr.pooled, k=40, l2=True)
    ev_pool = ctx.h.knn(tr.pooled, ev.pooled, k=40, l2=True)

    def robust_scale(train_values, values):
        center = np.median(train_values)
        scale = np.median(np.abs(train_values - center)) * 1.4826
        if not np.isfinite(scale) or scale < 1e-8:
            scale = np.std(train_values)
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        return (values - center) / scale

    compact = robust_scale(tr_feat, ev_feat)
    angular = robust_scale(tr_pool, ev_pool)
    scores = 0.55 * compact + 0.45 * angular

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
