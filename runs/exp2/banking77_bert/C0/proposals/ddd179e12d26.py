import numpy as np

FAMILY = "logit"
ARM = "edl"
RATIONALE = (
    "This measures whether the relative shape of an item's Dirichlet concentration vector "
    "matches the in-scope class it most resembles, while explicitly removing total evidence."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8

    a_tr = np.maximum(ctx.h.alpha(tr.logits), eps)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), eps)

    # Centered log-concentrations retain the Dirichlet shape while removing
    # the overall evidence scale that drives vacuity and energy-like scores.
    z_tr = np.log(a_tr)
    z_tr = z_tr - z_tr.mean(axis=1, keepdims=True)
    z_ev = np.log(a_ev)
    z_ev = z_ev - z_ev.mean(axis=1, keepdims=True)

    k = a_tr.shape[1]
    global_mean = z_tr.mean(axis=0)
    global_var = z_tr.var(axis=0) + 1e-4
    means = np.tile(global_mean, (k, 1))
    variances = np.tile(global_var, (k, 1))

    for c in range(k):
        xc = z_tr[tr.labels == c]
        if xc.shape[0] >= 2:
            means[c] = xc.mean(axis=0)
            variances[c] = xc.var(axis=0) + 1e-4
        elif xc.shape[0] == 1:
            means[c] = xc[0]
            variances[c] = global_var

    predicted = np.argmax(a_ev, axis=1)
    delta = z_ev - means[predicted]
    shape_surprise = np.mean(
        (delta * delta) / variances[predicted],
        axis=1,
    )

    return np.nan_to_num(shape_surprise, nan=0.0, posinf=1e12, neginf=0.0)
