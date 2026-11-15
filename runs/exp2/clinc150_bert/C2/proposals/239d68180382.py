import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Average distance to several nearby normalized pooled encoder embeddings is less sensitive to isolated training points than a single k-th-neighbor distance. OOS queries should lie farther from their local in-scope semantic neighborhood."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_x = ctx.h.l2(tr.pooled)
    eval_x = ctx.h.l2(ev.pooled)
    d8 = ctx.h.knn(train_x, eval_x, k=8, l2=False)
    d24 = ctx.h.knn(train_x, eval_x, k=24, l2=False)
    d64 = ctx.h.knn(train_x, eval_x, k=64, l2=False)
    scores = 0.25 * d8 + 0.35 * d24 + 0.40 * d64
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
