import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A multiscale cosine k-nearest-neighbour radius estimates whether an utterance lies in a locally supported in-scope encoder region. Combining small and larger neighbourhood radii reduces sensitivity to duplicate examples while retaining sensitivity to sparse OOS regions."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_x = ctx.h.l2(tr.pooled)
    eval_x = ctx.h.l2(ev.pooled)

    d3 = ctx.h.knn(train_x, eval_x, k=3, l2=False)
    d15 = ctx.h.knn(train_x, eval_x, k=15, l2=False)
    d60 = ctx.h.knn(train_x, eval_x, k=60, l2=False)

    scores = (
        0.45 * np.log1p(d3)
        + 0.35 * np.log1p(d15)
        + 0.20 * np.log1p(d60)
    )
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
