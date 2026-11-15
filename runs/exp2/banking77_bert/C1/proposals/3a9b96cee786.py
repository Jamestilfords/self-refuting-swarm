import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Multi-scale nearest-neighbour distances in both the bottleneck and encoder spaces detect queries that are locally unsupported by training intents. Combining normalized representations reduces sensitivity to a single embedding geometry."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    d_feat = ctx.h.knn(tr.feats, ev.feats, k=15, l2=True)
    d_pool = ctx.h.knn(tr.pooled, ev.pooled, k=35, l2=True)

    out = 0.45 * d_feat + 0.55 * d_pool
    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)
