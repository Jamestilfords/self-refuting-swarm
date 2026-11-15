import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-specific shrunk diagonal variances preserve low-variance intent constraints that a tied covariance washes out, while shrinkage keeps estimates stable for smaller classes."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    n, d = x.shape
    c = int(np.max(y)) + 1
    counts = np.bincount(y, minlength=c).astype(float)

    sums = np.zeros((c, d), dtype=float)
    np.add.at(sums, y, x)
    means = sums / np.maximum(counts[:, None], 1.0)

    sq_sums = np.zeros((c, d), dtype=float)
    np.add.at(sq_sums, y, x * x)
    raw_var = sq_sums / np.maximum(counts[:, None], 1.0) - means * means
    raw_var = np.maximum(raw_var, 0.0)

    global_mean = np.mean(x, axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0)
    floor = np.maximum(np.median(global_var) * 1e-5, 1e-8)
    global_var = np.maximum(global_var, floor)

    shrink = 32.0 / (counts + 32.0)
    variances = (1.0 - shrink[:, None]) * raw_var + shrink[:, None] * global_var
    variances = np.maximum(variances, floor)

    log_prior = np.log(np.maximum(counts / float(n), 1e-12))
    log_norm = -0.5 * (d * np.log(2.0 * np.pi) + np.sum(np.log(variances), axis=1))

    log_mix = np.full(z.shape[0], -np.inf, dtype=float)
    for cls in range(c):
        if counts[cls] <= 0:
            continue
        diff = z - means[cls]
        quad = np.sum((diff * diff) / variances[cls], axis=1)
        component = log_prior[cls] + log_norm[cls] - 0.5 * quad
        log_mix = np.logaddexp(log_mix, component)

    return np.maximum(-log_mix, 0.0)
