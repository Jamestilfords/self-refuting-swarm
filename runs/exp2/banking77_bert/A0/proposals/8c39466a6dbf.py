import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A classifier-assigned intent should support the query under its own per-feature variance profile, "
    "with empirical-Bayes shrinkage stabilizing diagonal covariances for small classes."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0009", "m_0021"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    k = tr.logits.shape[1]
    d = x.shape[1]
    global_mean = np.mean(x, axis=0)
    global_var = np.var(x, axis=0) + 1e-6

    means = np.empty((k, d), dtype=float)
    variances = np.empty((k, d), dtype=float)

    for c in range(k):
        xc = x[y == c]
        n_c = xc.shape[0]
        if n_c == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            means[c] = np.mean(xc, axis=0)
            class_var = np.var(xc, axis=0) if n_c > 1 else global_var
            shrink = d / (d + max(n_c - 1, 1))
            variances[c] = (1.0 - shrink) * class_var + shrink * global_var
            variances[c] = np.maximum(variances[c], 1e-6)

    pred = np.argmax(ev.logits, axis=1)
    residual = z - means[pred]
    dist = np.sum((residual * residual) / variances[pred], axis=1)

    return np.log1p(dist / float(d))
