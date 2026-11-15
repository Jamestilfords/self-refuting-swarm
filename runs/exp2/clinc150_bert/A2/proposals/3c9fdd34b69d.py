import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The centered log-ratio shape of Dirichlet concentrations captures relative evidence allocation "
    "across intents while removing total evidential mass. In-scope queries should match the class-specific "
    "evidence shape learned from training data, whereas OOS queries can have atypical concentration profiles."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0103"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8

    a_tr = np.maximum(ctx.h.alpha(tr.logits), eps)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), eps)

    x_tr = np.log(a_tr)
    x_tr = x_tr - x_tr.mean(axis=1, keepdims=True)
    x_ev = np.log(a_ev)
    x_ev = x_ev - x_ev.mean(axis=1, keepdims=True)

    n_classes = tr.logits.shape[1]
    dim = x_tr.shape[1]
    global_mean = x_tr.mean(axis=0)
    global_var = x_tr.var(axis=0) + 1e-5

    means = np.empty((n_classes, dim), dtype=float)
    variances = np.empty((n_classes, dim), dtype=float)
    for c in range(n_classes):
        xc = x_tr[tr.labels == c]
        if len(xc) >= 2:
            means[c] = xc.mean(axis=0)
            variances[c] = xc.var(axis=0) + 1e-5
        elif len(xc) == 1:
            means[c] = xc[0]
            variances[c] = global_var
        else:
            means[c] = global_mean
            variances[c] = global_var

    pred = np.argmax(a_ev, axis=1)
    residual = (x_ev - means[pred]) ** 2 / variances[pred]
    return np.mean(np.minimum(residual, 100.0), axis=1)
