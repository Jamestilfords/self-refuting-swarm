import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "A moderately large k-nearest-neighbour radius in L2-normalized pooled representation "
    "space measures support from the broader in-scope semantic manifold rather than only a single near duplicate."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    return np.asarray(ctx.h.knn(tr.pooled, ev.pooled, k=50, l2=True), dtype=float)
