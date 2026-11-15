import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Neighborhood distance in both the compact bottleneck and the encoder-pooled semantic space requires an input to be locally supported by in-scope examples under two complementary representations. Combining train-calibrated distances changes the ranking from a single-space k-nearest-neighbor detector."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_feat = ctx.h.knn(tr.feats, tr.feats, k=16, l2=True)
    train_pool = ctx.h.knn(tr.pooled, tr.pooled, k=8, l2=True)

    feat_med = np.median(train_feat)
    pool_med = np.median(train_pool)
    feat_scale = np.median(np.abs(train_feat - feat_med)) + 1e-8
    pool_scale = np.median(np.abs(train_pool - pool_med)) + 1e-8

    eval_feat = ctx.h.knn(tr.feats, ev.feats, k=16, l2=True)
    eval_pool = ctx.h.knn(tr.pooled, ev.pooled, k=8, l2=True)

    z_feat = (eval_feat - feat_med) / feat_scale
    z_pool = (eval_pool - pool_med) / pool_scale
    return np.log1p(np.exp(np.clip(z_feat, -30.0, 30.0))) + np.log1p(
        np.exp(np.clip(z_pool, -30.0, 30.0))
    )
