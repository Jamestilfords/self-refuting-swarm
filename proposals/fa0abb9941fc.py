import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Averaging normalized kNN radii across several neighborhood scales in encoder space measures both immediate and broader in-scope support. OOS queries should remain distant from training examples at all scales rather than only lacking one very close neighbor."
SOURCE = "https://proceedings.mlr.press/v162/sun22d/sun22d.pdf"
PARENTS = []
MEMORY_USED = ["m_0055", "m_0123"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    r5 = ctx.h.knn(tr.pooled, ev.pooled, k=5, l2=True)
    r25 = ctx.h.knn(tr.pooled, ev.pooled, k=25, l2=True)
    r100 = ctx.h.knn(tr.pooled, ev.pooled, k=100, l2=True)
    scores = 0.50 * r5 + 0.30 * r25 + 0.20 * r100
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
