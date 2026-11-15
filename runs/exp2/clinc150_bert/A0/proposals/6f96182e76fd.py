import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Averaging several local neighbor distances in L2-normalized pooled space estimates semantic support more stably than a single-neighbor radius. OOS queries should have consistently weak support across a moderate local neighborhood."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0111"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_x = ctx.h.l2(tr.pooled)
    eval_x = ctx.h.l2(ev.pooled)
    radii = [
        ctx.h.knn(train_x, eval_x, k=k, l2=False)
        for k in (5, 10, 20, 40)
    ]
    score = np.mean(np.stack(radii, axis=0), axis=0)
    return np.nan_to_num(score, nan=0.0, posinf=1e12, neginf=-1e12)
