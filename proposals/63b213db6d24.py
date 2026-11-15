import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A classifier-selected intent should have support under its own coordinate-wise feature "
    "variation, not merely under a covariance shared by all intents. Shrinking each class's "
    "diagonal variance toward the global variance stabilizes rare intents while preserving "
    "intent-specific constrained directions."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013", "m_0066"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    k = tr.logits.shape[1]
    d = x.shape[1]
    eps = 1e-8

    global_mean = x.mean(axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0) + eps

    means = np.empty((k, d), dtype=np.float64)
    variances = np.empty((k, d), dtype=np.float64)
    offsets = np.empty(k, dtype=np.float64)

    for c in range(k):
        xc = x[y == c]
        n_c = xc.shape[0]
        if n_c == 0:
            means[c] = global_mean
            variances[c] = global_var
            offsets[c] = 0.0
            continue

        mu = xc.mean(axis=0)
        raw_var = np.mean((xc - mu) ** 2, axis=0)
        shrink = d / (d + n_c)
        var = (1.0 - shrink) * raw_var + shrink * global_var + eps

        means[c] = mu
        variances[c] = var

        train_nll = 0.5 * (
            np.sum((xc - mu) ** 2 / var, axis=1) + np.sum(np.log(var))
        )
        offsets[c] = np.median(train_nll)

    pred = np.argmax(ev.logits, axis=1)
    mu = means[pred]
    var = variances[pred]
    nll = 0.5 * (np.sum((z - mu) ** 2 / var, axis=1) + np.sum(np.log(var), axis=1))

    return nll - offsets[pred]
