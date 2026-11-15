import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-specific diagonal covariances retain intent-dependent feature scales that a tied covariance discards. Shrinking each class variance toward the global variance stabilizes estimates for small classes while penalizing samples that do not fit any intent's feature profile."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels)

    classes = np.unique(y)
    d = x.shape[1]
    global_var = np.var(x, axis=0) + 1e-6
    scores = np.full(z.shape[0], np.inf, dtype=np.float64)

    for c in classes:
        xc = x[y == c]
        n = xc.shape[0]
        mean = np.mean(xc, axis=0)
        if n > 1:
            class_var = np.var(xc, axis=0)
        else:
            class_var = global_var

        shrink = n / (n + 20.0)
        var = shrink * class_var + (1.0 - shrink) * global_var
        var = np.maximum(var, 1e-6)

        diff = z - mean
        nll = np.sum((diff * diff) / var, axis=1) + np.sum(np.log(var))
        scores = np.minimum(scores, nll)

    return scores
