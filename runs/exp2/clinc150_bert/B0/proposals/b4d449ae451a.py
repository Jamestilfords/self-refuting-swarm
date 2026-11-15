import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Class-specific diagonal variances retain intent-dependent feature geometry that a tied covariance "
    "can wash out, while shrinkage toward the global variance keeps small intents stable."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=float)
    x_eval = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels)

    global_var = np.var(x_train, axis=0) + 1e-6
    classes = np.unique(y)
    best = np.full(x_eval.shape[0], np.inf, dtype=float)

    for c in classes:
        xc = x_train[y == c]
        n_c = xc.shape[0]
        mean_c = np.mean(xc, axis=0)
        var_c = np.var(xc, axis=0)
        shrink = n_c / (n_c + 20.0)
        var_c = shrink * var_c + (1.0 - shrink) * global_var
        var_c = np.maximum(var_c, 1e-6)

        diff = x_eval - mean_c
        nll = 0.5 * (
            np.sum((diff * diff) / var_c, axis=1) + np.sum(np.log(var_c))
        )
        best = np.minimum(best, nll)

    return np.nan_to_num(best, nan=0.0, posinf=1e12, neginf=-1e12)
