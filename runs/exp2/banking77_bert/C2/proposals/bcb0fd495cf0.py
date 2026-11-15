import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Averaging distances to several nearby normalized encoder representations reduces sensitivity to isolated training examples while identifying queries outside local in-scope neighborhoods."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_x = ctx.h.l2(tr.pooled)
    eval_x = ctx.h.l2(ev.pooled)
    k_values = (5, 15, 35)
    distances = [ctx.h.knn(train_x, eval_x, k=k, l2=False) for k in k_values]
    return np.mean(np.stack(distances, axis=1), axis=1)
