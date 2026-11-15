import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Averaging several nearby normalized encoder-neighbor distances is less sensitive to duplicate training examples than a single kth-neighbor distance. OOS queries should have consistently larger local angular distances to the in-scope training manifold."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)
    return (
        0.20 * ctx.h.knn(x_train, x_eval, k=5, l2=False)
        + 0.30 * ctx.h.knn(x_train, x_eval, k=15, l2=False)
        + 0.50 * ctx.h.knn(x_train, x_eval, k=35, l2=False)
    )
