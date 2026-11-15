import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "The mean distance to several nearby normalized encoder representations measures local in-distribution support more robustly than a single nearest-neighbor distance. OOS queries should lie farther from their local in-scope neighborhood."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)
    k = min(25, len(x_train))
    if k < 1:
        return np.arange(len(x_eval), dtype=float)
    return ctx.h.knn(x_train, x_eval, k=k, l2=False)
