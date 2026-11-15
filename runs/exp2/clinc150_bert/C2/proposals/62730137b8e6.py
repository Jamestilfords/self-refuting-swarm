import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A multi-scale cosine kNN radius in the bottleneck space measures both local support and broader neighborhood support, making isolated OOS queries score highly."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    d5 = ctx.h.knn(tr.feats, ev.feats, k=5, l2=True)
    d20 = ctx.h.knn(tr.feats, ev.feats, k=20, l2=True)
    d64 = ctx.h.knn(tr.feats, ev.feats, k=64, l2=True)
    return np.log1p((0.45 * d5) + (0.35 * d20) + (0.20 * d64))
