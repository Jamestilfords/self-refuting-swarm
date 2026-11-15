import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A multi-scale nearest-neighbour score in both bottleneck and encoder spaces measures whether a query lacks nearby in-scope support at local and broader neighbourhood scales."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    pooled_scores = (
        0.20 * ctx.h.knn(tr.pooled, ev.pooled, k=5, l2=True)
        + 0.30 * ctx.h.knn(tr.pooled, ev.pooled, k=15, l2=True)
        + 0.50 * ctx.h.knn(tr.pooled, ev.pooled, k=40, l2=True)
    )
    feat_scores = (
        0.20 * ctx.h.knn(tr.feats, ev.feats, k=5, l2=True)
        + 0.30 * ctx.h.knn(tr.feats, ev.feats, k=15, l2=True)
        + 0.50 * ctx.h.knn(tr.feats, ev.feats, k=40, l2=True)
    )

    scores = 0.45 * pooled_scores + 0.55 * feat_scores
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
