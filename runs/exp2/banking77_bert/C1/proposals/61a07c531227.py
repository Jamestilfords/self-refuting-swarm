FAMILY = "knn"
ARM = "ce"
RATIONALE = "OOS queries are expected to lie farther from the in-scope training manifold in normalized encoder space. Using a larger neighborhood than the standard nearest-neighbor setting makes the score less sensitive to isolated training examples."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    return ctx.h.knn(tr.pooled, ev.pooled, k=50, l2=True)
