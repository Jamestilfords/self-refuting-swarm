import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-specific diagonal variances retain intent-dependent feature spread that a tied covariance discards. Shrinking each class variance toward the global diagonal covariance keeps estimates stable for small classes while exposing atypical OOS feature patterns."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels)

    classes, inverse = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    n, d = x.shape

    global_mean = np.mean(x, axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0)
    global_var = np.maximum(global_var, 1e-8)

    counts = np.bincount(inverse, minlength=n_classes).astype(float)
    means = np.zeros((n_classes, d), dtype=float)
    second = np.zeros((n_classes, d), dtype=float)
    np.add.at(means, inverse, x)
    np.add.at(second, inverse, x * x)
    means /= counts[:, None]

    class_var = second / counts[:, None] - means * means
    class_var = np.maximum(class_var, 0.0)

    # Small classes rely more heavily on the globally estimated diagonal scale.
    weight = counts / (counts + 20.0)
    variances = weight[:, None] * class_var + (1.0 - weight[:, None]) * global_var
    variances = np.maximum(variances, 1e-8)

    inv_var = 1.0 / variances
    log_det = np.sum(np.log(variances), axis=1)
    log_prior = np.log(np.maximum(counts / float(n), 1e-12))

    z_sq = z * z
    quadratic = (
        z_sq @ inv_var.T
        - 2.0 * z @ (means * inv_var).T
        + np.sum(means * means * inv_var, axis=1)
    )
    nll = 0.5 * (quadratic + log_det[None, :]) - log_prior[None, :]
    return np.min(nll, axis=1)
