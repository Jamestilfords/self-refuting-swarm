import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The centered log-ratio of Dirichlet concentrations captures how evidential mass is "
    "distributed among intents while removing total-evidence magnitude. OOS inputs can have "
    "an atypical predicted-intent evidence composition even when vacuity or energy is normal."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0041"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    a_tr = np.maximum(ctx.h.alpha(tr.logits), eps)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), eps)

    # Centered log-ratio representation removes the overall concentration scale.
    z_tr = np.log(a_tr)
    z_tr = z_tr - z_tr.mean(axis=1, keepdims=True)
    z_ev = np.log(a_ev)
    z_ev = z_ev - z_ev.mean(axis=1, keepdims=True)

    y = np.asarray(tr.labels, dtype=np.int64)
    k = z_tr.shape[1]
    global_mean = z_tr.mean(axis=0)
    global_var = z_tr.var(axis=0)
    var_floor = np.maximum(global_var * 1e-3, 1e-6)

    means = np.empty((k, k), dtype=float)
    variances = np.empty((k, k), dtype=float)
    for c in range(k):
        members = z_tr[y == c]
        n_c = members.shape[0]
        if n_c == 0:
            means[c] = global_mean
            variances[c] = global_var + var_floor
        else:
            means[c] = members.mean(axis=0)
            if n_c > 1:
                class_var = members.var(axis=0)
            else:
                class_var = global_var
            # Shrink sparse-class estimates toward the global composition variance.
            weight = n_c / (n_c + 10.0)
            variances[c] = weight * class_var + (1.0 - weight) * global_var
            variances[c] = np.maximum(variances[c], var_floor)

    pred = np.argmax(ev.logits, axis=1)
    delta = z_ev - means[pred]
    score_values = np.mean((delta * delta) / variances[pred], axis=1)
    return np.nan_to_num(score_values, nan=0.0, posinf=1e12, neginf=0.0)
