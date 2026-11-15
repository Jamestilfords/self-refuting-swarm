import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Local support in a PCA-compressed pooled encoder space captures semantic novelty while suppressing noisy high-dimensional directions. The 30th normalized neighbour distance requires sustained support from known in-scope examples rather than one unusually close match."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0051"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    pca = ctx.h.pca(tr.pooled, dim=96)
    train_z = ctx.h.l2(pca.transform(tr.pooled))
    eval_z = ctx.h.l2(pca.transform(ev.pooled))
    return ctx.h.knn(train_z, eval_z, k=30, l2=False)
