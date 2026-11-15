import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A diagonal class-conditional model with sample-size shrinkage captures class-specific feature dispersion that a tied covariance misses. OOS inputs should have low likelihood under every shrunk class distribution."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels)

    classes = np.unique(y)
    d = x.shape[1]
    global_mean = np.mean(x, axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0)
    floor = 1e-6 * (np.mean(global_var) + 1e-12)
    global_var = np.maximum(global_var, floor)

    means = np.empty((len(classes), d), dtype=float)
    variances = np.empty((len(classes), d), dtype=float)

    for i, c in enumerate(classes):
        xc = x[y == c]
        n = xc.shape[0]
        mu = np.mean(xc, axis=0)
        if n > 1:
            class_var = np.mean((xc - mu) ** 2, axis=0)
        else:
            class_var = global_var
        shrink = n / (n + 20.0)
        means[i] = mu
        variances[i] = np.maximum(
            shrink * class_var + (1.0 - shrink) * global_var,
            floor,
        )

    inv_var = 1.0 / variances
    mean_inv = means * inv_var
    constants = (
        np.sum(means * mean_inv, axis=1)
        + np.sum(np.log(variances), axis=1)
        + d * np.log(2.0 * np.pi)
    )

    quadratic = (
        (z * z) @ inv_var.T
        - 2.0 * z @ mean_inv.T
        + constants[None, :]
    )
    return 0.5 * np.min(quadratic, axis=1)
