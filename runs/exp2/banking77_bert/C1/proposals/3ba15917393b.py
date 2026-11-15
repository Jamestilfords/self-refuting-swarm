import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-specific diagonal variances capture anisotropic in-scope feature clusters that a tied covariance misses. Shrinking each class variance toward the global variance keeps estimates stable for small classes."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels)

    global_var = np.var(x_train, axis=0) + 1e-6
    classes = np.unique(labels)
    best = np.full(x_eval.shape[0], np.inf, dtype=np.float64)

    for c in classes:
        xc = x_train[labels == c]
        n_c = xc.shape[0]
        mean_c = np.mean(xc, axis=0)
        if n_c > 1:
            var_c = np.var(xc, axis=0)
        else:
            var_c = global_var

        shrink = 20.0 / (n_c + 20.0)
        var_c = (1.0 - shrink) * var_c + shrink * global_var
        var_c = np.maximum(var_c, 1e-6)

        log_det = np.sum(np.log(var_c))
        dist = np.zeros(x_eval.shape[0], dtype=np.float64)
        for start in range(0, x_eval.shape[1], 128):
            stop = min(start + 128, x_eval.shape[1])
            delta = x_eval[:, start:stop] - mean_c[start:stop]
            dist += np.sum(
                (delta * delta) / var_c[start:stop],
                axis=1,
            )

        nll = 0.5 * (dist + log_det)
        best = np.minimum(best, nll)

    return best
