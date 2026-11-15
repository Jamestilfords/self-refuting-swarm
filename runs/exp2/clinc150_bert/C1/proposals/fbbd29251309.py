import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A moderately large k-NN radius in L2-normalized encoder space estimates local in-scope support more robustly than a single nearest neighbour. OOS queries should require a larger radius to reach many semantically related training examples."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)
    k = min(50, x_train.shape[0])
    return ctx.h.knn(x_train, x_eval, k=k, l2=False)
