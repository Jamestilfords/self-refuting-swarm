import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Averaging neighborhood radii at two train-fitted scales in normalized encoder space detects samples that lack local in-scope support while reducing sensitivity to an individual neighbor."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)

    d10_train = ctx.h.knn(x_train, x_train, k=11, l2=False)
    d50_train = ctx.h.knn(x_train, x_train, k=51, l2=False)
    d10_eval = ctx.h.knn(x_train, x_eval, k=10, l2=False)
    d50_eval = ctx.h.knn(x_train, x_eval, k=50, l2=False)

    finite10 = d10_train[np.isfinite(d10_train)]
    finite50 = d50_train[np.isfinite(d50_train)]
    scale10 = max(float(np.median(finite10)), 1e-8) if finite10.size else 1.0
    scale50 = max(float(np.median(finite50)), 1e-8) if finite50.size else 1.0

    scores = 0.5 * (d10_eval / scale10) + 0.5 * (d50_eval / scale50)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
