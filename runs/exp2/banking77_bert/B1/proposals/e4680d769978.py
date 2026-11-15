import numpy as np

FAMILY = "logit"
ARM = "edl"
RATIONALE = (
    "The centered log-ratio shape of Dirichlet concentrations captures how evidence is "
    "distributed among intents while removing total concentration (vacuity). OOS queries "
    "may have an atypical evidence-allocation pattern for their predicted intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-8)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-8)

    # Centered log-ratio coordinates retain concentration shape but discard scale.
    z_tr = np.log(a_tr)
    z_tr = z_tr - z_tr.mean(axis=1, keepdims=True)
    z_ev = np.log(a_ev)
    z_ev = z_ev - z_ev.mean(axis=1, keepdims=True)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]
    d = z_tr.shape[1]

    global_mean = z_tr.mean(axis=0)
    global_var = z_tr.var(axis=0)
    var_floor = np.maximum(np.median(global_var) * 1e-3, 1e-6)

    means = np.empty((k, d), dtype=float)
    variances = np.empty((k, d), dtype=float)

    for c in range(k):
        subset = z_tr[pred_tr == c]
        if subset.shape[0] >= 4:
            means[c] = subset.mean(axis=0)
            # Shrink unstable classwise variances toward global variation.
            raw_var = subset.var(axis=0)
            weight = subset.shape[0] / (subset.shape[0] + 12.0)
            variances[c] = weight * raw_var + (1.0 - weight) * global_var
        else:
            means[c] = global_mean
            variances[c] = global_var

    variances = np.maximum(variances, var_floor)
    delta = z_ev - means[pred_ev]
    return np.mean((delta * delta) / variances[pred_ev], axis=1)
