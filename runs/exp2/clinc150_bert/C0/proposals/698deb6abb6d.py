import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "A query is suspicious when the classifier assigns probability to classes whose class-specific, shrinkage-regularized diagonal feature distributions all fit poorly. Using the softmax-weighted residual avoids reducing the method to a nearest-class Gaussian density score."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    classes = np.unique(y)
    dim = x.shape[1]
    global_mean = np.mean(x, axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0)
    global_var = np.maximum(global_var, 1e-6)

    probs = ctx.h.softmax(ev.logits)
    if probs.shape[1] != len(classes):
        classes = np.arange(probs.shape[1])

    expected_residual = np.zeros(z.shape[0], dtype=float)
    total_weight = np.zeros(z.shape[0], dtype=float)

    for c in classes:
        mask = y == c
        nc = int(np.sum(mask))
        if nc == 0:
            continue

        xc = x[mask]
        mean_c = np.mean(xc, axis=0)
        if nc > 1:
            var_c = np.mean((xc - mean_c) ** 2, axis=0)
        else:
            var_c = global_var

        shrink = nc / (nc + 20.0)
        var_c = shrink * var_c + (1.0 - shrink) * global_var
        var_c = np.maximum(var_c, 1e-6)

        q = np.sum((z - mean_c) ** 2 / var_c, axis=1) / float(dim)
        if c < probs.shape[1]:
            w = probs[:, c]
            expected_residual += w * q
            total_weight += w

    return expected_residual / np.maximum(total_weight, 1e-12)
