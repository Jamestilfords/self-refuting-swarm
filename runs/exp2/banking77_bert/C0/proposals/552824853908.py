import numpy as np

FAMILY = "density"
ARM = "edl"
RATIONALE = "Class-conditional centered log-Dirichlet concentration profiles capture whether the relative evidence pattern resembles any in-scope intent, while removing total concentration so the score is not a monotone transform of energy or vacuity."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-8)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-8)

    z_tr = np.log(a_tr)
    z_ev = np.log(a_ev)
    z_tr = z_tr - z_tr.mean(axis=1, keepdims=True)
    z_ev = z_ev - z_ev.mean(axis=1, keepdims=True)

    global_var = np.var(z_tr, axis=0) + 1e-4
    classes = np.unique(tr.labels)
    best = np.full(z_ev.shape[0], np.inf)

    for c in classes:
        mask = tr.labels == c
        xc = z_tr[mask]
        mean = xc.mean(axis=0)
        if xc.shape[0] > 1:
            class_var = np.var(xc, axis=0)
            weight = xc.shape[0] / (xc.shape[0] + 10.0)
            var = weight * class_var + (1.0 - weight) * global_var
        else:
            var = global_var

        dist = np.sum((z_ev - mean) ** 2 / (var + 1e-5), axis=1)
        best = np.minimum(best, dist)

    return np.nan_to_num(best, nan=0.0, posinf=1e12, neginf=0.0)
