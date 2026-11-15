import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "The mean distance to several nearby L2-normalized encoder representations measures local support rather than relying on a single nearest-neighbour match. OOS queries should lie in lower-density neighbourhoods and have larger average neighbour distances."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)
    k = min(25, len(x_train))
    if k < 1:
        return np.zeros(len(x_eval), dtype=float)
    return ctx.h.knn(x_train, x_eval, k=k, l2=False)
