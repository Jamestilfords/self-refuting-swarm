FAMILY = "knn"
ARM = "ce"
RATIONALE = "L2-normalized bottleneck features emphasize intent-discriminative local geometry, while a moderately large kNN radius reduces sensitivity to isolated in-scope examples. Queries outside known intent neighborhoods should have larger cosine-neighborhood distance."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0008"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    return ctx.h.knn(tr.feats, ev.feats, k=50, l2=True)
