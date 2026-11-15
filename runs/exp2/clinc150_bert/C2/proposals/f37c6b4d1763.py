FAMILY = "knn"
ARM = "ce"
RATIONALE = "OOS queries should lie farther from their local in-scope neighborhood in L2-normalized bottleneck space. Using the 25th neighbor emphasizes local support rather than sensitivity to a single unusually close training example."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = min(25, tr.feats.shape[0])
    return ctx.h.knn(tr.feats, ev.feats, k=k, l2=True)
