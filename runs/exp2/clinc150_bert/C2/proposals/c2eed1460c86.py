import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-specific diagonal variances retain directions that are stable for one intent but variable for another, unlike a tied covariance. Shrinkage toward the global variance stabilizes small classes while preserving heteroscedastic intent geometry."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    classes, inverse = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    n, d = x.shape

    counts = np.bincount(inverse, minlength=n_classes).astype(float)
    means = np.zeros((n_classes, d), dtype=float)
    np.add.at(means, inverse, x)
    means /= counts[:, None]

    global_mean = x.mean(axis=0)
    global_var = ((x - global_mean) ** 2).mean(axis=0)
    floor = max(float(np.mean(global_var)) * 1e-6, 1e-10)
    global_var = np.maximum(global_var, floor)

    variances = np.empty((n_classes, d), dtype=float)
    for c in range(n_classes):
        xc = x[inverse == c]
        if len(xc) > 1:
            empirical = ((xc - means[c]) ** 2).mean(axis=0)
        else:
            empirical = global_var
        shrink = d / (counts[c] + d)
        variances[c] = (1.0 - shrink) * empirical + shrink * global_var

    variances = np.maximum(variances, floor)
    precision = 1.0 / variances
    logdet = np.log(variances).sum(axis=1)
    log_prior = np.log(np.maximum(counts / float(n), 1e-12))

    quadratic = (
        (z * z) @ precision.T
        - 2.0 * z @ (means * precision).T
        + ((means * means) * precision).sum(axis=1)[None, :]
    )
    nll = 0.5 * (quadratic + logdet[None, :]) - log_prior[None, :]
    return np.minimum(nll.min(axis=1), 1e12)
