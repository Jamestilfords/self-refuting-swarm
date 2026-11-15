import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "OOS queries should lie in persistently sparse neighborhoods of the normalized encoder representation. Averaging several neighbor radii reduces sensitivity to isolated duplicates or a single locally dense training region."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    d8 = ctx.h.knn(tr.pooled, ev.pooled, k=8, l2=True)
    d24 = ctx.h.knn(tr.pooled, ev.pooled, k=24, l2=True)
    d48 = ctx.h.knn(tr.pooled, ev.pooled, k=48, l2=True)
    return (d8 + d24 + d48) / 3.0
