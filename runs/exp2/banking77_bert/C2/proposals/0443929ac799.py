import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A moderately large k-NN radius in L2-normalized encoder space measures whether a query lies within a locally supported in-scope semantic neighborhood. Averaging several neighbor radii reduces sensitivity to isolated training points while retaining local density information."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)
    k_values = (15, 30, 60)
    distances = [ctx.h.knn(x_train, x_eval, k=k, l2=False) for k in k_values]
    return np.mean(np.stack(distances, axis=1), axis=1)
