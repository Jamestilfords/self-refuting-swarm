import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Averaging distances to several nearby normalized encoder representations reduces sensitivity to individual training outliers while identifying queries outside local in-scope neighborhoods."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)
    dists = []
    for k in (5, 10, 20, 40):
        dists.append(ctx.h.knn(x_train, x_eval, k=k, l2=False))
    return np.mean(np.column_stack(dists), axis=1)
