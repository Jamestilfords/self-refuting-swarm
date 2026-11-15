import numpy as np
from scipy.special import logsumexp

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Class-specific shrunk diagonal covariances preserve intent-dependent feature variability "
    "while avoiding unstable full QDA estimates. Low mixture density flags queries unsupported "
    "by every known intent, including those near a mean only along high-variance dimensions."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0077"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    classes, inverse = np.unique(y, return_inverse=True)
    n_classes = classes.size
    dim = x.shape[1]
    n_train = x.shape[0]

    global_mean = x.mean(axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0)
    floor = max(float(np.mean(global_var)) * 1e-6, 1e-10)
    global_var = np.maximum(global_var, floor)

    log_terms = np.empty((z.shape[0], n_classes), dtype=np.float64)
    log_2pi = np.log(2.0 * np.pi)

    for j in range(n_classes):
        xc = x[inverse == j]
        count = xc.shape[0]
        mean = xc.mean(axis=0)

        if count > 1:
            class_var = np.mean((xc - mean) ** 2, axis=0)
        else:
            class_var = global_var.copy()

        shrink = count / (count + 20.0)
        var = shrink * class_var + (1.0 - shrink) * global_var
        var = np.maximum(var, floor)

        diff = z - mean
        mahal = np.sum((diff * diff) / var, axis=1)
        log_det = np.sum(np.log(var))
        log_prior = np.log(count / n_train)
        log_terms[:, j] = log_prior - 0.5 * (mahal + log_det + dim * log_2pi)

    return -logsumexp(log_terms, axis=1)
