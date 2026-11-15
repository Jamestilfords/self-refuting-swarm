import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A mixture of short- and medium-range normalized encoder-neighborhood distances measures both immediate local support and broader in-scope manifold coverage. OOS queries should be farther from training examples at both neighborhood scales."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    pca = ctx.h.pca(tr.pooled, 256)
    train_x = pca.transform(tr.pooled)
    eval_x = pca.transform(ev.pooled)
    near = ctx.h.knn(train_x, eval_x, k=8, l2=True)
    broad = ctx.h.knn(train_x, eval_x, k=32, l2=True)
    return np.nan_to_num(0.6 * near + 0.4 * broad, nan=0.0, posinf=1e12, neginf=-1e12)
