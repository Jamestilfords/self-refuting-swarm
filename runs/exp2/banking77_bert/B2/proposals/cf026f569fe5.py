import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Averaging several neighborhood radii in L2-normalized encoder space measures local in-scope support more stably than a single k-th-neighbor distance. OOS queries should remain distant across both fine and moderately broad training neighborhoods."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0133"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)
    d5 = ctx.h.knn(x_train, x_eval, k=5, l2=False)
    d15 = ctx.h.knn(x_train, x_eval, k=15, l2=False)
    d45 = ctx.h.knn(x_train, x_eval, k=45, l2=False)
    return (d5 + d15 + d45) / 3.0
