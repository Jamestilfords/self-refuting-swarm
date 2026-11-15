import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Averaging several k-nearest-neighbour radii in L2-normalized encoder space measures support across local and moderately broader in-scope neighborhoods. OOS queries should lie farther from training examples consistently across these scales."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    radii = [
        ctx.h.knn(tr.pooled, ev.pooled, k=5, l2=True),
        ctx.h.knn(tr.pooled, ev.pooled, k=15, l2=True),
        ctx.h.knn(tr.pooled, ev.pooled, k=45, l2=True),
    ]
    out = (radii[0] + radii[1] + radii[2]) / 3.0
    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)
