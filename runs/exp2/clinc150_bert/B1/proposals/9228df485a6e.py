import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A multiscale normalized-neighbor score combines absolute semantic isolation at a wider neighborhood scale with local neighborhood expansion. OOS queries should be far from in-scope examples and often have less compact normalized pooled-feature neighborhoods."
SOURCE = "https://proceedings.mlr.press/v162/sun22d/sun22d.pdf"
PARENTS = []
MEMORY_USED = ["m_0089"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    d4 = ctx.h.knn(tr.pooled, ev.pooled, k=4, l2=True)
    d32 = ctx.h.knn(tr.pooled, ev.pooled, k=32, l2=True)
    score = np.log1p(np.maximum(d32, 0.0)) + 0.35 * np.log1p(
        np.maximum(d32, 0.0) / (np.maximum(d4, 0.0) + 1e-8)
    )
    return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
