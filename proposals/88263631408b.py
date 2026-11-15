import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Intent-specific diagonal variances identify feature coordinates that are normally tightly "
    "constrained for the classifier-selected intent. A soft aggregate of unusually large "
    "coordinate-wise standardized residuals detects distributed support violations without "
    "collapsing them into a single Gaussian density."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013", "m_0027"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    k = tr.logits.shape[1]
    d = x.shape[1]
    shrink = 12.0

    global_mean = np.mean(x, axis=0)
    global_var = np.var(x, axis=0)
    floor = np.maximum(1e-8, 1e-4 * np.mean(global_var))
    global_var = np.maximum(global_var, floor)

    counts = np.bincount(y, minlength=k).astype(float)
    means = np.empty((k, d), dtype=float)
    variances = np.empty((k, d), dtype=float)

    for c in range(k):
        n = counts[c]
        if n > 0:
            xc = x[y == c]
            class_mean = np.mean(xc, axis=0)
            class_var = np.var(xc, axis=0)
            means[c] = (n * class_mean + shrink * global_mean) / (n + shrink)
            variances[c] = (
                n * class_var + shrink * global_var
            ) / (n + shrink)
        else:
            means[c] = global_mean
            variances[c] = global_var

    variances = np.maximum(variances, floor)
    pred = np.argmax(ev.logits, axis=1)

    out = np.empty(z.shape[0], dtype=float)
    threshold = 9.0  # squared three-standard-deviation residual
    for c in np.unique(pred):
        idx = pred == c
        diff = z[idx] - means[c]
        r2 = (diff * diff) / variances[c]
        r2 = np.nan_to_num(r2, nan=0.0, posinf=1e12, neginf=0.0)
        excess = r2 - threshold
        soft_excess = np.maximum(excess, 0.0) + np.log1p(
            np.exp(-np.abs(excess))
        )
        out[idx] = np.mean(soft_excess, axis=1)

    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=0.0)
