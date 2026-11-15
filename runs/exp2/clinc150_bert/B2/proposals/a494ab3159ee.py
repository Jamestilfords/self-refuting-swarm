FAMILY = "knn"
ARM = "ce"
RATIONALE = "A moderately large k-nearest-neighbour distance in L2-normalized encoder space measures whether a query lies in a locally supported in-scope region, rather than only matching one unusually close training example."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    return ctx.h.knn(tr.pooled, ev.pooled, k=32, l2=True)
