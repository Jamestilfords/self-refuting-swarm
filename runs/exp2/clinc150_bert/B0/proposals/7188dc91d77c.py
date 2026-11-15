FAMILY = "knn"
ARM = "ce"
RATIONALE = "A moderately large neighborhood in L2-normalized pooled encoder space estimates whether a query lies within a locally supported in-scope semantic region. OOS queries are expected to have a larger distance to their 50th nearest in-scope neighbor than isolated but valid near-duplicate examples."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    return ctx.h.knn(tr.pooled, ev.pooled, k=50, l2=True)
