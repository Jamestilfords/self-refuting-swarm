import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Averaging several local neighbour distances in normalized pooled space is less sensitive to an anomalously close single training example than a single k-th-neighbour distance. OOS queries should lie in sparser semantic neighbourhoods than in-scope queries."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)
    d10 = ctx.h.knn(x_train, x_eval, k=10, l2=False)
    d30 = ctx.h.knn(x_train, x_eval, k=30, l2=False)
    d80 = ctx.h.knn(x_train, x_eval, k=80, l2=False)
    return (d10 + d30 + d80) / 3.0
