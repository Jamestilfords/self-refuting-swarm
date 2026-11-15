import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Intent classes can occupy manifolds with different feature-wise spreads, so a shrunk "
    "class-conditional diagonal Gaussian penalizes deviations that a tied covariance can miss."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels)

    classes, inverse = np.unique(y, return_inverse=True)
    n_classes = classes.size
    dim = x.shape[1]

    counts = np.bincount(inverse, minlength=n_classes).astype(np.float64)
    means = np.zeros((n_classes, dim), dtype=np.float64)
    np.add.at(means, inverse, x)
    means /= np.maximum(counts[:, None], 1.0)

    centered = x - means[inverse]
    class_var = np.zeros((n_classes, dim), dtype=np.float64)
    np.add.at(class_var, inverse, centered * centered)
    class_var /= np.maximum(counts[:, None], 1.0)

    global_var = np.var(x, axis=0)
    floor = np.maximum(global_var * 1e-4, 1e-8)
    shrink = counts / (counts + 20.0)
    variances = shrink[:, None] * class_var + (1.0 - shrink[:, None]) * global_var
    variances = np.maximum(variances, floor)

    inv_var = 1.0 / variances
    mean_inv = means * inv_var
    constants = (
        np.sum(means * mean_inv, axis=1)
        + np.sum(np.log(variances), axis=1)
        - 2.0 * np.log(np.maximum(counts / counts.sum(), 1e-12))
    )

    quadratic = (
        (z * z) @ inv_var.T
        - 2.0 * (z @ mean_inv.T)
        + constants[None, :]
    )
    return 0.5 * np.min(quadratic, axis=1)
