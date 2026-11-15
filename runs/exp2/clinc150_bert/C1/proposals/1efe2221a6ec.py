import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "This score measures whether the centered log-shape of an item's Dirichlet concentration vector matches any class-specific evidential signature learned on training data. It uses relative concentration geometry rather than total evidence alone, so it is not a monotone transform of energy or vacuity."
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

    labels = tr.labels.astype(int)
    classes = np.unique(labels)
    global_var = np.var(z_tr, axis=0) + 1e-4
    best = np.full(z_ev.shape[0], np.inf, dtype=float)

    for c in classes:
        xc = z_tr[labels == c]
        mu = xc.mean(axis=0)
        if xc.shape[0] > 1:
            var = np.var(xc, axis=0)
            var = np.maximum(var, 0.1 * global_var) + 1e-4
        else:
            var = global_var

        diff = z_ev - mu
        dist = np.sum((diff * diff) / var, axis=1)
        best = np.minimum(best, dist)

    strength = np.log(np.sum(a_ev, axis=1))
    train_strength = np.log(np.sum(a_tr, axis=1))
    center = np.median(train_strength)
    scale = np.median(np.abs(train_strength - center)) + 1e-4

    return best + 0.05 * np.abs(strength - center) / scale
