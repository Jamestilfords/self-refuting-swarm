import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Multi-scale k-nearest-neighbour distances on both bottleneck and encoder features "
    "measure local in-distribution support while reducing sensitivity to a single neighbourhood size."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0115"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    d5_feat = ctx.h.knn(tr.feats, ev.feats, k=5, l2=True)
    d25_feat = ctx.h.knn(tr.feats, ev.feats, k=25, l2=True)
    d5_pool = ctx.h.knn(tr.pooled, ev.pooled, k=5, l2=True)
    d25_pool = ctx.h.knn(tr.pooled, ev.pooled, k=25, l2=True)

    distances = np.stack([d5_feat, d25_feat, d5_pool, d25_pool], axis=1)
    distances = np.maximum(np.nan_to_num(distances, nan=0.0, posinf=1e12, neginf=0.0), 0.0)
    return np.mean(np.log1p(distances), axis=1)
