import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Averaging distances to several nearby normalized encoder representations is less sensitive to an individual mislabeled or atypical training point than a single-neighbor distance. OOS queries should have consistently larger local distances in the semantic encoder space."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)

    ks = (5, 10, 20, 40)
    distances = [ctx.h.knn(x_train, x_eval, k=k, l2=False) for k in ks]
    score_values = np.mean(np.stack(distances, axis=1), axis=1)

    if np.all(score_values == score_values[0]):
        score_values = np.linalg.norm(x_eval - x_train.mean(axis=0), axis=1)
    return np.nan_to_num(score_values, nan=0.0, posinf=1e12, neginf=-1e12)
