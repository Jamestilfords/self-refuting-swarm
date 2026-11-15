import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A moderately large neighborhood in L2-normalized encoder space estimates local in-scope support while reducing sensitivity to isolated training outliers."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    return ctx.h.knn(tr.pooled, ev.pooled, k=35, l2=True)
