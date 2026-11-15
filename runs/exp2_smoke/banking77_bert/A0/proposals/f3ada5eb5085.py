FAMILY = "knn"
ARM = "ce"
RATIONALE = "A moderately large k-nearest-neighbour radius in L2-normalized bottleneck space measures whether an item lies in a locally supported in-scope semantic region. Using k=32 reduces sensitivity to isolated training examples while retaining sensitivity to sparse near-OOS regions."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    return ctx.h.knn(tr.feats, ev.feats, k=32, l2=True)
