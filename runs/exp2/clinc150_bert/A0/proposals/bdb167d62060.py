import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A query should be locally supported both in the encoder's semantic pooled space and in the classifier bottleneck space. Combining train-normalized neighborhood radii across these complementary representations highlights examples that lack support in either geometry."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    k = 30
    tr_pool = ctx.h.knn(tr.pooled, tr.pooled, k=k + 1, l2=True)
    tr_feat = ctx.h.knn(tr.feats, tr.feats, k=k + 1, l2=True)

    ev_pool = ctx.h.knn(tr.pooled, ev.pooled, k=k, l2=True)
    ev_feat = ctx.h.knn(tr.feats, ev.feats, k=k, l2=True)

    pool_scale = np.median(tr_pool[tr_pool > 0])
    feat_scale = np.median(tr_feat[tr_feat > 0])
    pool_scale = max(float(pool_scale), 1e-12)
    feat_scale = max(float(feat_scale), 1e-12)

    a = np.maximum(ev_pool / pool_scale, 1e-12)
    b = np.maximum(ev_feat / feat_scale, 1e-12)
    return np.sqrt(a * b)
