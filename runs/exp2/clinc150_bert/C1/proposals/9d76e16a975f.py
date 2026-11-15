import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Averaging several local neighbor distances in L2-normalized encoder space is less sensitive to isolated training outliers than a single kth-neighbor distance. OOS queries should lie farther from their local in-scope neighborhood."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)

    d8 = ctx.h.knn(x_train, x_eval, k=8, l2=False)
    d24 = ctx.h.knn(x_train, x_eval, k=24, l2=False)
    d64 = ctx.h.knn(x_train, x_eval, k=64, l2=False)

    score = 0.25 * d8 + 0.35 * d24 + 0.40 * d64
    return np.asarray(score, dtype=float)
