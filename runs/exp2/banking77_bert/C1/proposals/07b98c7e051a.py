import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "This measures whether the relative shape of an item's Dirichlet concentration vector matches the concentration shapes seen for its predicted in-scope class. It ignores overall concentration scale, so it is not a monotone transform of energy or vacuity."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    alpha_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-8)
    alpha_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-8)

    # Log-ratio coordinates retain the Dirichlet shape while removing
    # the common concentration scale associated with vacuity.
    z_tr = np.log(alpha_tr)
    z_tr = z_tr - z_tr.mean(axis=1, keepdims=True)
    z_ev = np.log(alpha_ev)
    z_ev = z_ev - z_ev.mean(axis=1, keepdims=True)

    labels = tr.labels.astype(int)
    n_classes = alpha_tr.shape[1]
    dim = z_tr.shape[1]

    global_center = z_tr.mean(axis=0)
    global_var = z_tr.var(axis=0) + 1e-4
    centers = np.tile(global_center, (n_classes, 1))
    counts = np.bincount(labels, minlength=n_classes)

    for c in range(n_classes):
        if counts[c] > 0:
            centers[c] = z_tr[labels == c].mean(axis=0)

    residuals = z_tr - centers[labels]
    pooled_var = residuals.var(axis=0) + 0.05 * global_var + 1e-4

    predicted = np.argmax(alpha_ev, axis=1)
    diff = z_ev - centers[predicted]
    return np.mean((diff * diff) / pooled_var, axis=1)
