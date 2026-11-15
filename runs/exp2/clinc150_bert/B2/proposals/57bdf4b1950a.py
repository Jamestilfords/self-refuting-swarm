import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Distance to a moderately sized neighborhood in the compact bottleneck feature space measures local in-scope support without relying on the high-dimensional pooled representation. OOS queries should require a larger unnormalized feature-space radius to reach comparable training support."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0107"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    d = ctx.h.knn(tr.feats, ev.feats, k=25, l2=False)
    return np.asarray(d, dtype=float)
