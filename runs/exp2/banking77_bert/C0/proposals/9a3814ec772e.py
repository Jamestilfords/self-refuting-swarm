FAMILY = "knn"
ARM = "ce"
RATIONALE = "A multi-scale kNN score combines normalized encoder-pooled and bottleneck representations, so queries that are locally unsupported in either semantic space receive higher OOS scores."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    pooled_dist = ctx.h.knn(tr.pooled, ev.pooled, k=31, l2=True)
    feat_dist = ctx.h.knn(tr.feats, ev.feats, k=17, l2=True)
    return 0.65 * pooled_dist + 0.35 * feat_dist
