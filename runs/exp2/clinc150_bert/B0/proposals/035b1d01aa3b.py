FAMILY = "knn"
ARM = "ce"
RATIONALE = "A multiscale cosine k-nearest-neighbour score detects queries that are locally isolated from training utterances while remaining robust to individual near-duplicate examples. Larger neighbourhoods capture broader intent-manifold departure."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    d5 = ctx.h.knn(tr.pooled, ev.pooled, k=5, l2=True)
    d25 = ctx.h.knn(tr.pooled, ev.pooled, k=25, l2=True)
    d75 = ctx.h.knn(tr.pooled, ev.pooled, k=75, l2=True)
    return 0.2 * d5 + 0.35 * d25 + 0.45 * d75
