import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-specific diagonal variances preserve intent-dependent feature spread that a tied covariance discards. Shrinkage toward the global diagonal covariance keeps rare-class estimates stable while rewarding queries that fit no class distribution."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    classes = np.unique(y)
    n_classes = classes.size
    dim = x.shape[1]

    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    means = np.zeros((n_classes, dim), dtype=np.float64)
    for c in range(n_classes):
        xc = x[y == c]
        if xc.shape[0]:
            means[c] = xc.mean(axis=0)

    global_var = x.var(axis=0) + 1e-6
    variances = np.empty((n_classes, dim), dtype=np.float64)
    shrink = 20.0
    for c in range(n_classes):
        xc = x[y == c]
        if xc.shape[0] > 1:
            local_var = xc.var(axis=0)
        else:
            local_var = global_var
        variances[c] = (
            counts[c] * local_var + shrink * global_var
        ) / (counts[c] + shrink)

    floor = 1e-5 * np.maximum(global_var, 1e-6)
    variances = np.maximum(variances, floor[None, :])

    inv_var = 1.0 / variances
    linear = means * inv_var
    constants = (
        np.sum(means * means * inv_var, axis=1)
        + np.sum(np.log(variances), axis=1)
    )

    out = np.empty(z.shape[0], dtype=np.float64)
    batch = 2048
    for start in range(0, z.shape[0], batch):
        zz = z[start:start + batch]
        quad = (zz * zz) @ inv_var.T
        cross = zz @ linear.T
        nll = 0.5 * (quad - 2.0 * cross + constants[None, :])
        out[start:start + zz.shape[0]] = np.min(nll, axis=1)

    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)
