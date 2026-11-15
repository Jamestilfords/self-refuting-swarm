import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A diagonal covariance per intent preserves intent-specific semantic tolerances while shrinkage toward the pooled within-intent variance stabilizes small classes. The minimum shrunk-QDA surprise detects queries unsupported by any known intent without assuming a single shared covariance."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0076"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    classes = np.unique(y)
    global_mean = np.mean(x, axis=0)
    centered = x - global_mean
    global_var = np.mean(centered * centered, axis=0)
    floor = max(float(np.mean(global_var)) * 1e-6, 1e-10)
    global_var = np.maximum(global_var, floor)

    best = np.full(z.shape[0], np.inf, dtype=np.float64)
    tau = 20.0

    for c in classes:
        xc = x[y == c]
        n = xc.shape[0]
        mean = np.mean(xc, axis=0)
        if n > 1:
            d = xc - mean
            class_var = np.mean(d * d, axis=0)
        else:
            class_var = global_var

        shrink = n / (n + tau)
        var = shrink * class_var + (1.0 - shrink) * global_var
        var = np.maximum(var, floor)

        diff = z - mean
        nll = 0.5 * (
            np.sum((diff * diff) / var, axis=1) + np.sum(np.log(var))
        )
        best = np.minimum(best, nll)

    return np.nan_to_num(best, nan=0.0, posinf=1e12, neginf=-1e12)
