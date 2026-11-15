import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Intent representations can have substantially different per-coordinate dispersion, so a "
    "regularized class-specific diagonal Gaussian assigns low support to queries unlike every intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0105"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels)

    classes, inv = np.unique(y, return_inverse=True)
    n_classes = classes.size
    dim = x.shape[1]

    global_mean = x.mean(axis=0)
    global_var = x.var(axis=0) + 1e-6
    means = np.empty((n_classes, dim), dtype=np.float64)
    variances = np.empty((n_classes, dim), dtype=np.float64)
    log_priors = np.empty(n_classes, dtype=np.float64)

    for c in range(n_classes):
        xc = x[inv == c]
        count = xc.shape[0]
        mean = xc.mean(axis=0)
        centered = xc - mean
        class_var = (centered * centered).mean(axis=0)
        shrink = dim / (dim + count)
        means[c] = mean
        variances[c] = (1.0 - shrink) * class_var + shrink * global_var + 1e-6
        log_priors[c] = np.log(count / x.shape[0])

    best = np.full(z.shape[0], -np.inf, dtype=np.float64)
    log2pi = np.log(2.0 * np.pi)
    for c in range(n_classes):
        diff = z - means[c]
        logp = (
            log_priors[c]
            - 0.5 * (
                np.sum((diff * diff) / variances[c], axis=1)
                + np.sum(np.log(variances[c]))
                + dim * log2pi
            )
        )
        best = np.maximum(best, logp)

    return -best
