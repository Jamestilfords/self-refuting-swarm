import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Multi-scale cosine kNN distances in a train-fitted low-rank pooled representation measure "
    "whether a query has local support across both dense and broader in-scope neighborhoods."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    pca = ctx.h.pca(tr.pooled, dim=96)
    train_z = ctx.h.l2(pca.transform(tr.pooled))
    eval_z = ctx.h.l2(pca.transform(ev.pooled))

    d5 = ctx.h.knn(train_z, eval_z, k=5, l2=False)
    d15 = ctx.h.knn(train_z, eval_z, k=15, l2=False)
    d35 = ctx.h.knn(train_z, eval_z, k=35, l2=False)
    scores = 0.50 * d5 + 0.30 * d15 + 0.20 * d35
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
