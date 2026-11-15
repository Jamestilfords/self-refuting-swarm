import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Averaging normalized encoder-space kNN radii across small, medium, and broad "
    "neighborhoods reduces sensitivity to accidental close neighbors while measuring "
    "whether a query remains supported at multiple local density scales."
)
SOURCE = "https://proceedings.mlr.press/v162/sun22d/sun22d.pdf"
PARENTS = []
MEMORY_USED = ["m_0055"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    d_near = ctx.h.knn(tr.pooled, ev.pooled, k=4, l2=True)
    d_mid = ctx.h.knn(tr.pooled, ev.pooled, k=20, l2=True)
    d_broad = ctx.h.knn(tr.pooled, ev.pooled, k=80, l2=True)

    scores = (np.log1p(d_near) + np.log1p(d_mid) + np.log1p(d_broad)) / 3.0
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
