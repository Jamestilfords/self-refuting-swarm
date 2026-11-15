import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Average distance to several nearby normalized encoder representations is less sensitive to an anomalous single neighbour than a single k-th-neighbour distance. OOS queries should lie farther from their local in-scope neighbourhood."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)
    d10 = ctx.h.knn(x_train, x_eval, k=10, l2=False)
    d30 = ctx.h.knn(x_train, x_eval, k=30, l2=False)
    d60 = ctx.h.knn(x_train, x_eval, k=60, l2=False)
    return (d10 + d30 + d60) / 3.0
