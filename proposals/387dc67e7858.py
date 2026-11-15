import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Averaging several L2-normalized pooled-embedding kNN radii measures support across local "
    "neighborhood scales, reducing sensitivity to an accidental close training neighbor."
)
SOURCE = "https://proceedings.mlr.press/v162/sun22d/sun22d.pdf"
PARENTS = []
MEMORY_USED = ["m_0055", "m_0123"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    radii = (
        ctx.h.knn(tr.pooled, ev.pooled, k=8, l2=True)
        + ctx.h.knn(tr.pooled, ev.pooled, k=24, l2=True)
        + ctx.h.knn(tr.pooled, ev.pooled, k=64, l2=True)
    ) / 3.0
    tie = np.tanh(np.asarray(ev.pooled, dtype=float)[:, 0])
    return np.asarray(radii, dtype=float) + 1e-8 * tie
