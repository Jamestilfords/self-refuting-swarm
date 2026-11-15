FAMILY = "knn"
ARM = "ce"
RATIONALE = "A multiscale nearest-neighbour distance in L2-normalized encoder space measures whether a query lies in a dense local in-scope neighborhood. Averaging several neighborhood radii is less sensitive to isolated training examples than a single nearest-neighbour rank."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    d5 = ctx.h.knn(tr.pooled, ev.pooled, k=5, l2=True)
    d20 = ctx.h.knn(tr.pooled, ev.pooled, k=20, l2=True)
    d80 = ctx.h.knn(tr.pooled, ev.pooled, k=80, l2=True)
    return 0.2 * d5 + 0.3 * d20 + 0.5 * d80
