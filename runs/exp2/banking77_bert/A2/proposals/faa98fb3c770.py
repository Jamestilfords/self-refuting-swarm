import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-specific diagonal variances retain dimensions that are selectively constrained by individual intents, unlike a tied covariance model. Shrinking each class variance toward the global variance stabilizes rare classes while preserving heterogeneous intent geometry."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels)

    classes, inverse = np.unique(y, return_inverse=True)
    c = classes.size
    d = x.shape[1]

    global_mean = np.mean(x, axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0)
    scale = max(float(np.mean(global_var)), 1e-12)
    floor = scale * 1e-8

    means = np.empty((c, d), dtype=np.float64)
    variances = np.empty((c, d), dtype=np.float64)
    counts = np.bincount(inverse, minlength=c).astype(np.float64)
    shrink_strength = 20.0

    for j in range(c):
        members = x[inverse == j]
        means[j] = np.mean(members, axis=0)
        centered = members - means[j]
        class_var = np.mean(centered * centered, axis=0)
        variances[j] = (
            counts[j] * class_var + shrink_strength * global_var
        ) / (counts[j] + shrink_strength)

    variances = np.maximum(variances, floor)
    precision = 1.0 / variances

    quadratic = (
        (z * z) @ precision.T
        - 2.0 * z @ (means * precision).T
        + np.sum(means * means * precision, axis=1)
    )
    nll = 0.5 * (quadratic + np.sum(np.log(variances), axis=1))
    log_prior = np.log(np.maximum(counts / np.sum(counts), 1e-12))
    return np.min(nll - log_prior[None, :], axis=1)
