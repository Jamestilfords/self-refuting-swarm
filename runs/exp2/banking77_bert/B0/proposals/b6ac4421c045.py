FAMILY = "knn"
ARM = "ce"
RATIONALE = "A moderately broad k-nearest-neighbour radius in PCA-compressed bottleneck space measures local support while suppressing encoder dimensions unrelated to intent structure. OOS queries should require a larger neighborhood radius even when their classifier confidence is high."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0137"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    pca = ctx.h.pca(tr.feats, 96)
    train_x = pca.transform(tr.feats)
    eval_x = pca.transform(ev.feats)
    return ctx.h.knn(train_x, eval_x, k=25, l2=True)
