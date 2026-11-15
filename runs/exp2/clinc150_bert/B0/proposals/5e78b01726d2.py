import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A multiscale normalized-feature kNN score averages local and broader neighborhood distances, making it less sensitive to isolated in-scope points than a single neighbor rank. OOS queries should remain distant across both neighborhood scales."
SOURCE = "https://proceedings.mlr.press/v162/sun22d/sun22d.pdf"
PARENTS = []
MEMORY_USED = ["m_0087"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    n = len(tr.labels)
    ks = sorted(set([max(1, min(5, n)), max(1, min(20, n)), max(1, min(80, n))]))
    distances = [ctx.h.knn(tr.pooled, ev.pooled, k=k, l2=True) for k in ks]
    return np.mean(np.stack(distances, axis=0), axis=0)
