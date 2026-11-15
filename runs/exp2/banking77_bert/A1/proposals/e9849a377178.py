import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Averaging several raw bottleneck-space kNN radii measures both immediate local support "
    "and broader within-intent neighborhood support without relying on evaluation-set calibration."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0117"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    d8 = ctx.h.knn(tr.feats, ev.feats, k=8, l2=False)
    d32 = ctx.h.knn(tr.feats, ev.feats, k=32, l2=False)
    d96 = ctx.h.knn(tr.feats, ev.feats, k=96, l2=False)
    scores = (d8 + d32 + d96) / 3.0
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
