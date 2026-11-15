import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "The average distance to several nearby normalized encoder representations is less sensitive to a single accidental neighbor than a kth-neighbor score. OOS queries should have a consistently more distant local in-scope neighborhood."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)
    k_values = (5, 15, 40)
    distances = [ctx.h.knn(x_train, x_eval, k=k, l2=False) for k in k_values]
    score_out = np.mean(np.stack(distances, axis=1), axis=1)
    return np.nan_to_num(score_out, nan=0.0, posinf=1e12, neginf=-1e12)
