import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Averaging several normalized pooled-feature neighborhood radii measures local support across both very close semantic matches and broader intent neighborhoods. OOS queries should require consistently larger radii than in-scope queries rather than differing only at one chosen k."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0058"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)

    d1 = ctx.h.knn(x_train, x_eval, k=1, l2=False)
    d8 = ctx.h.knn(x_train, x_eval, k=8, l2=False)
    d32 = ctx.h.knn(x_train, x_eval, k=32, l2=False)
    d64 = ctx.h.knn(x_train, x_eval, k=64, l2=False)

    scores = 0.10 * d1 + 0.20 * d8 + 0.30 * d32 + 0.40 * d64
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
