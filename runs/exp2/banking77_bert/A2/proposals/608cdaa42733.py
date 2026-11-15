import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Multi-scale neighborhood distances in L2-normalized encoder space measure both immediate "
    "local support and broader manifold support. OOS queries should be farther from in-scope "
    "examples at both scales, while the geometric mean reduces sensitivity to isolated neighbors."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0027"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    d_near = ctx.h.knn(tr.pooled, ev.pooled, k=5, l2=True)
    d_broad = ctx.h.knn(tr.pooled, ev.pooled, k=32, l2=True)
    score = np.sqrt(np.maximum(d_near, 0.0) * np.maximum(d_broad, 0.0))
    return np.nan_to_num(score, nan=0.0, posinf=1e12, neginf=0.0)
