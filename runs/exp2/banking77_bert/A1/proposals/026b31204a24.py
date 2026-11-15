import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The centered log-ratio shape of Dirichlet concentrations captures relative evidential allocation "
    "across intents independently of total evidence. OOS queries may have an atypical pattern of competing "
    "class evidence even when their vacuity or energy appears in-distribution."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0085"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-8)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-8)

    # Centered log-ratios remove the overall Dirichlet evidence scale.
    z_tr = np.log(a_tr)
    z_tr = z_tr - z_tr.mean(axis=1, keepdims=True)
    z_ev = np.log(a_ev)
    z_ev = z_ev - z_ev.mean(axis=1, keepdims=True)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]
    dim = z_tr.shape[1]

    global_mean = z_tr.mean(axis=0)
    global_var = z_tr.var(axis=0) + 1e-4
    means = np.empty((n_classes, dim), dtype=float)
    variances = np.empty((n_classes, dim), dtype=float)

    for c in range(n_classes):
        members = z_tr[pred_tr == c]
        if len(members) >= 2:
            means[c] = members.mean(axis=0)
            # Shrink class variances toward global values for rare predicted classes.
            weight = len(members) / (len(members) + 12.0)
            variances[c] = weight * members.var(axis=0) + (1.0 - weight) * global_var
        else:
            means[c] = global_mean
            variances[c] = global_var

    residual = z_ev - means[pred_ev]
    score = np.mean((residual * residual) / (variances[pred_ev] + 1e-6), axis=1)
    return np.nan_to_num(score, nan=0.0, posinf=1e12, neginf=0.0)
