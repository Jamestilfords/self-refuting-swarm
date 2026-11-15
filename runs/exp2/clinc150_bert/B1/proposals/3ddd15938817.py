import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Multi-scale nearest-neighbor distances in normalized pooled embedding space measure both very local intent support and broader in-scope manifold support. OOS queries should remain distant at both neighborhood scales."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_x = ctx.h.l2(tr.pooled)
    eval_x = ctx.h.l2(ev.pooled)

    local = ctx.h.knn(train_x, eval_x, k=8, l2=False)
    broad = ctx.h.knn(train_x, eval_x, k=48, l2=False)

    scores = 0.65 * np.asarray(local, dtype=float) + 0.35 * np.asarray(broad, dtype=float)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
