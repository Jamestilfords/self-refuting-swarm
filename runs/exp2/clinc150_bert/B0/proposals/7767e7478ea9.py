import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Multi-scale neighborhood distances capture both very local support and broader intent-manifold support. "
    "OOS queries should remain isolated across several neighborhood scales rather than only at one chosen k."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_x = ctx.h.l2(tr.pooled)
    eval_x = ctx.h.l2(ev.pooled)

    d8 = ctx.h.knn(train_x, eval_x, k=8, l2=False)
    d32 = ctx.h.knn(train_x, eval_x, k=32, l2=False)
    d96 = ctx.h.knn(train_x, eval_x, k=96, l2=False)

    scores = (
        0.25 * np.log1p(np.maximum(d8, 0.0))
        + 0.35 * np.log1p(np.maximum(d32, 0.0))
        + 0.40 * np.log1p(np.maximum(d96, 0.0))
    )
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
