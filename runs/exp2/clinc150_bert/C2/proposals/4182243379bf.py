import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A moderately large k-NN radius in L2-normalized pooled encoder space measures whether a query lies outside the local support of in-scope training utterances. Averaging several nearby-neighbor radii is less sensitive to isolated training outliers than a single nearest-neighbor distance."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)
    d8 = ctx.h.knn(x_train, x_eval, k=8, l2=False)
    d24 = ctx.h.knn(x_train, x_eval, k=24, l2=False)
    d48 = ctx.h.knn(x_train, x_eval, k=48, l2=False)
    scores = 0.2 * d8 + 0.35 * d24 + 0.45 * d48
    return np.maximum(scores, 0.0)
