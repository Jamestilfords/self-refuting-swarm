FAMILY = "knn"
ARM = "ce"
RATIONALE = "L2-normalized bottleneck features emphasize semantic direction rather than activation magnitude. The 25th-neighbor distance measures whether a query lies in a locally supported in-scope region while being less sensitive than a single-neighbor score."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    return ctx.h.knn(tr.feats, ev.feats, k=25, l2=True)
