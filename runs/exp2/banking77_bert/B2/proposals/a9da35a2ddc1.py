import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A larger neighbourhood radius in L2-normalized encoder space measures whether a query lies in a sparse region beyond local in-scope support. Using the 50th neighbour is less sensitive than a single-neighbour score to duplicates and isolated training points."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    scores = ctx.h.knn(tr.pooled, ev.pooled, k=50, l2=True)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
