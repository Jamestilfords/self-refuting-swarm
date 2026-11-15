import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A multiscale normalized kNN radius averages very local, neighborhood, and broader in-scope support. OOS queries tend to be far from training examples consistently across these scales."
SOURCE = "https://proceedings.mlr.press/v162/sun22d/sun22d.pdf"
PARENTS = []
MEMORY_USED = ["m_0086", "m_0087"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    d5 = ctx.h.knn(tr.pooled, ev.pooled, k=5, l2=True)
    d20 = ctx.h.knn(tr.pooled, ev.pooled, k=20, l2=True)
    d80 = ctx.h.knn(tr.pooled, ev.pooled, k=80, l2=True)
    scores = 0.25 * d5 + 0.35 * d20 + 0.40 * d80
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
