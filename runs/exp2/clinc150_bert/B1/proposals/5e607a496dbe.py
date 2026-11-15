import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A multi-scale kNN radius on L2-normalized bottleneck features estimates both immediate neighbourhood support and broader in-scope manifold density. OOS queries should require larger radii across these train-defined neighbourhood scales."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    d8 = ctx.h.knn(tr.feats, ev.feats, k=8, l2=True)
    d32 = ctx.h.knn(tr.feats, ev.feats, k=32, l2=True)
    d96 = ctx.h.knn(tr.feats, ev.feats, k=96, l2=True)
    scores = 0.25 * d8 + 0.35 * d32 + 0.40 * d96
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
