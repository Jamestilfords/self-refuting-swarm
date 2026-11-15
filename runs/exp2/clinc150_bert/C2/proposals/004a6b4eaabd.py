import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "The mean distance to several normalized encoder neighbors measures local support rather than relying on a single nearest boundary point. OOS queries should have consistently weaker neighborhood support in pooled representation space."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)
    k_values = (5, 15, 35)
    distances = [ctx.h.knn(x_train, x_eval, k=k, l2=False) for k in k_values]
    return np.mean(np.column_stack(distances), axis=1)
