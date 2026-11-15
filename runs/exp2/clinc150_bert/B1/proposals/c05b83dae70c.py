import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Multi-scale angular kNN isolation in pooled encoder space detects queries that lack both close paraphrase-level neighbors and broader semantic in-scope support. Combining several neighborhood radii is less sensitive to a single choice of k."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0070"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_x = ctx.h.l2(tr.pooled)
    eval_x = ctx.h.l2(ev.pooled)

    d8 = ctx.h.knn(train_x, eval_x, k=8, l2=False)
    d32 = ctx.h.knn(train_x, eval_x, k=32, l2=False)
    d128 = ctx.h.knn(train_x, eval_x, k=128, l2=False)

    eps = 1e-12
    return np.log(np.maximum(d8, eps)) * 0.25 + np.log(
        np.maximum(d32, eps)
    ) * 0.50 + np.log(np.maximum(d128, eps)) * 0.25
