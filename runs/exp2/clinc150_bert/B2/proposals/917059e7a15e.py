import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Class-specific diagonal covariance models retain intent-dependent feature scales that a tied covariance "
    "averages away. Shrinking each class variance toward the global variance stabilizes rare intents while "
    "making deviations along tightly constrained class dimensions strongly OOS-like."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0010"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.pooled, dtype=np.float64)

    k = tr.logits.shape[1]
    d = x.shape[1]
    global_mean = np.mean(x, axis=0)
    global_var = np.var(x, axis=0)
    floor = np.maximum(1e-8, 1e-6 * np.mean(global_var) + 1e-12)

    means = np.empty((k, d), dtype=np.float64)
    variances = np.empty((k, d), dtype=np.float64)

    for c in range(k):
        xc = x[y == c]
        n = xc.shape[0]
        if n == 0:
            means[c] = global_mean
            variances[c] = global_var + floor
        else:
            mc = np.mean(xc, axis=0)
            vc = np.var(xc, axis=0)
            shrink = n / (n + 20.0)
            means[c] = mc
            variances[c] = shrink * vc + (1.0 - shrink) * global_var + floor

    pred = np.argmax(ev.logits, axis=1)
    mu = means[pred]
    var = variances[pred]

    standardized_sq = np.sum((z - mu) ** 2 / var, axis=1)
    log_volume = np.sum(np.log(var), axis=1)

    # A class-conditioned shrunk diagonal negative log density.
    return 0.5 * (standardized_sq + log_volume)
