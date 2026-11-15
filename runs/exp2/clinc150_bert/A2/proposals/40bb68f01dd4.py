FAMILY = "knn"
ARM = "ce"
RATIONALE = "A multiscale nearest-neighbour distance combines local novelty with broader training-manifold sparsity. OOS queries should be distant from both their closest examples and moderately sized neighborhoods of in-scope training queries."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    d5 = ctx.h.knn(tr.pooled, ev.pooled, k=5, l2=True)
    d15 = ctx.h.knn(tr.pooled, ev.pooled, k=15, l2=True)
    d40 = ctx.h.knn(tr.pooled, ev.pooled, k=40, l2=True)
    return 0.20 * d5 + 0.35 * d15 + 0.45 * d40
