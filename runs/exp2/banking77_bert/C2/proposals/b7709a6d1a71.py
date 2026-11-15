import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A moderately large k-NN radius in L2-normalized pooled encoder space estimates local support rather than relying on a single nearest neighbour. OOS queries should require a larger neighborhood radius to reach many in-scope training examples."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)
    return ctx.h.knn(x_train, x_eval, k=25, l2=False)
