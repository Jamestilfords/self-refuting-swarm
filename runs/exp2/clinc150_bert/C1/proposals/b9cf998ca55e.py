import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A multiscale nearest-neighbour distance in normalized encoder space measures both immediate local support and broader in-scope manifold coverage. OOS queries should tend to be far from training examples at both neighbourhood scales."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    d_near = ctx.h.knn(tr.pooled, ev.pooled, k=8, l2=True)
    d_far = ctx.h.knn(tr.pooled, ev.pooled, k=64, l2=True)
    scores = 0.35 * d_near + 0.65 * d_far
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
