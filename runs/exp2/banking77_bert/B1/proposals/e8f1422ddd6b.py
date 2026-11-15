import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "OOS queries should lack nearby support in both the bottleneck representation and the encoder pooled space. Combining local radii across these complementary feature spaces rewards agreement rather than relying on a single kNN geometry."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    tr_feat = ctx.h.l2(tr.feats)
    ev_feat = ctx.h.l2(ev.feats)
    tr_pool = ctx.h.l2(tr.pooled)
    ev_pool = ctx.h.l2(ev.pooled)

    d_feat = ctx.h.knn(tr_feat, ev_feat, k=16, l2=False)
    d_pool = ctx.h.knn(tr_pool, ev_pool, k=16, l2=False)

    scores = np.log1p(np.maximum(d_feat, 0.0)) + np.log1p(np.maximum(d_pool, 0.0))
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
