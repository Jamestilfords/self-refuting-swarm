import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-specific diagonal variances capture intent-dependent feature spread that a tied covariance misses. Shrinkage toward the global variance stabilizes small classes while retaining anisotropic per-class support."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    classes, inverse = np.unique(y, return_inverse=True)
    n_classes = classes.size
    dim = x.shape[1]

    global_mean = np.mean(x, axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0)
    global_var = np.maximum(global_var, 1e-8)

    best = np.full(z.shape[0], np.inf, dtype=np.float64)
    for c in range(n_classes):
        xc = x[inverse == c]
        count = xc.shape[0]
        mean = np.mean(xc, axis=0)

        if count > 1:
            var = np.mean((xc - mean) ** 2, axis=0)
        else:
            var = global_var.copy()

        shrink = dim / (dim + max(count - 1, 1))
        var = (1.0 - shrink) * var + shrink * global_var
        var = np.maximum(var, 1e-8)

        logdet = np.sum(np.log(var))
        prior_penalty = -2.0 * np.log((count + 1.0) / (x.shape[0] + n_classes))
        dist = np.sum(((z - mean) ** 2) / var, axis=1) + logdet + prior_penalty
        best = np.minimum(best, dist)

    return np.nan_to_num(best, nan=0.0, posinf=1e12, neginf=-1e12)
