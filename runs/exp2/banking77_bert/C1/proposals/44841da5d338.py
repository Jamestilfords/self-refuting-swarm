import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Averaging distances to several nearby normalized encoder representations reduces sensitivity to isolated training points while preserving local support estimation. OOS queries should have consistently larger neighborhood distances than in-scope queries."
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
    return 0.2 * d8 + 0.35 * d24 + 0.45 * d48
