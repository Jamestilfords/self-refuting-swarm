import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "The mean distance to several nearby L2-normalized encoder representations estimates local in-scope support more robustly than a single neighbor. OOS queries should lie farther from their local training neighborhood."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)
    # Fixed random projection preserves local neighborhoods while reducing
    # memory and distance-computation cost for high-dimensional embeddings.
    rng = np.random.default_rng(271828)
    projection = rng.standard_normal((x_train.shape[1], 192))
    projection /= np.sqrt(192.0)
    x_train = ctx.h.l2(x_train @ projection)
    x_eval = ctx.h.l2(x_eval @ projection)
    return ctx.h.knn(x_train, x_eval, k=25, l2=False)
