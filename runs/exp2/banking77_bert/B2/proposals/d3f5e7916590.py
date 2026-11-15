import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "EDL outliers can have a Dirichlet concentration shape that is inconsistent with the "
    "class-specific evidence profile of their predicted intent, even when total evidence is similar."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-8)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-8)

    p_tr = a_tr / np.sum(a_tr, axis=1, keepdims=True)
    p_ev = a_ev / np.sum(a_ev, axis=1, keepdims=True)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]

    global_mean = np.mean(p_tr, axis=0)
    global_var = np.var(p_tr, axis=0) + 1e-6

    means = np.empty((k, p_tr.shape[1]), dtype=float)
    variances = np.empty_like(means)

    for c in range(k):
        mask = pred_tr == c
        if np.any(mask):
            pc = p_tr[mask]
            means[c] = np.mean(pc, axis=0)
            variances[c] = np.var(pc, axis=0) + 0.05 * global_var + 1e-6
        else:
            means[c] = global_mean
            variances[c] = global_var

    mu = means[pred_ev]
    var = variances[pred_ev]

    centered = p_ev - mu
    shape_distance = np.mean((centered * centered) / var, axis=1)

    entropy_ev = -np.sum(p_ev * np.log(np.maximum(p_ev, 1e-12)), axis=1)
    entropy_tr = -np.sum(p_tr * np.log(np.maximum(p_tr, 1e-12)), axis=1)
    entropy_mean = np.empty(k, dtype=float)
    entropy_var = np.empty(k, dtype=float)

    for c in range(k):
        mask = pred_tr == c
        if np.any(mask):
            values = entropy_tr[mask]
            entropy_mean[c] = np.mean(values)
            entropy_var[c] = np.var(values) + 1e-6
        else:
            entropy_mean[c] = np.mean(entropy_tr)
            entropy_var[c] = np.var(entropy_tr) + 1e-6

    entropy_deviation = (
        (entropy_ev - entropy_mean[pred_ev]) ** 2 / entropy_var[pred_ev]
    )

    return shape_distance + 0.15 * entropy_deviation
