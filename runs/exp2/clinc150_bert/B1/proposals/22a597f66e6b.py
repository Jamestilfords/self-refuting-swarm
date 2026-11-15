import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The relative allocation of Dirichlet concentration across intents can be atypical even when total "
    "evidence is high. Class-conditional deviations of centered log-concentration profiles flag such OOS shapes."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-8)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-8)

    z_tr = np.log(a_tr)
    z_tr = z_tr - z_tr.mean(axis=1, keepdims=True)
    z_ev = np.log(a_ev)
    z_ev = z_ev - z_ev.mean(axis=1, keepdims=True)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]
    d = z_tr.shape[1]

    global_mean = z_tr.mean(axis=0)
    global_var = z_tr.var(axis=0) + 1e-6
    means = np.empty((k, d), dtype=float)
    variances = np.empty((k, d), dtype=float)

    for c in range(k):
        xc = z_tr[pred_tr == c]
        if xc.shape[0] >= 4:
            means[c] = xc.mean(axis=0)
            variances[c] = xc.var(axis=0) + 1e-6
        else:
            means[c] = global_mean
            variances[c] = global_var

    diff = z_ev - means[pred_ev]
    return np.mean((diff * diff) / variances[pred_ev], axis=1)
