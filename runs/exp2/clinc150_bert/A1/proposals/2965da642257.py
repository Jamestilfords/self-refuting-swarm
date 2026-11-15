import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "A multiview local-neighborhood score averages angular distance in pooled encoder space "
    "and bottleneck space, so OOS inputs are flagged when they lack close in-scope semantic neighbors in either representation."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    pooled_train = ctx.h.l2(tr.pooled)
    pooled_eval = ctx.h.l2(ev.pooled)
    feat_train = ctx.h.l2(tr.feats)
    feat_eval = ctx.h.l2(ev.feats)

    d_pooled = ctx.h.knn(pooled_train, pooled_eval, k=32, l2=False)
    d_feat = ctx.h.knn(feat_train, feat_eval, k=32, l2=False)

    return 0.5 * d_pooled + 0.5 * d_feat
