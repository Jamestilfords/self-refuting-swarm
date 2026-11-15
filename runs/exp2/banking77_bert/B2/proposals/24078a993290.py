import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Known intents can have substantially different feature-wise dispersion, which a tied covariance "
    "misses. A predicted-intent-conditioned diagonal residual with variance shrinkage flags queries "
    "that violate the coordinate-wise constraints of the intent selected by the classifier."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0085"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    global_mean = x.mean(axis=0)
    global_var = x.var(axis=0)
    global_var = np.maximum(global_var, 1e-8)

    means = np.tile(global_mean, (k, 1))
    variances = np.tile(global_var, (k, 1))
    counts = np.bincount(y, minlength=k).astype(np.float64)

    for c in range(k):
        mask = y == c
        nc = int(mask.sum())
        if nc == 0:
            continue
        xc = x[mask]
        mu = xc.mean(axis=0)
        var = ((xc - mu) ** 2).mean(axis=0)
        # Shrink unstable intent-specific variances toward the global estimate.
        weight = nc / (nc + 24.0)
        means[c] = mu
        variances[c] = weight * var + (1.0 - weight) * global_var

    variances = np.maximum(variances, 1e-8)
    pred = np.argmax(ev.logits, axis=1)
    mu = means[pred]
    var = variances[pred]

    residual = z - mu
    squared = np.sum((residual * residual) / var, axis=1)
    # Normalize by dimension so score scale remains stable across bottleneck choices.
    return np.sqrt(np.maximum(squared / float(d), 0.0))
