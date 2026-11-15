import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Averaging normalized pooled-feature neighborhood radii across short, medium, and broad "
    "local scales makes the score less sensitive to a single neighbor while detecting queries "
    "that lack consistent semantic support from the in-scope training manifold."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)

    d8 = ctx.h.knn(x_train, x_eval, k=8, l2=False)
    d24 = ctx.h.knn(x_train, x_eval, k=24, l2=False)
    d64 = ctx.h.knn(x_train, x_eval, k=64, l2=False)

    score = 0.45 * d8 + 0.35 * d24 + 0.20 * d64
    return np.nan_to_num(score, nan=0.0, posinf=1e12, neginf=-1e12)
