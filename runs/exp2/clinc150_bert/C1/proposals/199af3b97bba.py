import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Averaging several local neighbour distances in L2-normalized encoder space reduces sensitivity to an unusually close single training example while detecting queries outside in-scope semantic neighborhoods."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)
    d10 = ctx.h.knn(x_train, x_eval, k=10, l2=False)
    d30 = ctx.h.knn(x_train, x_eval, k=30, l2=False)
    d75 = ctx.h.knn(x_train, x_eval, k=75, l2=False)
    return (d10 + d30 + d75) / 3.0
