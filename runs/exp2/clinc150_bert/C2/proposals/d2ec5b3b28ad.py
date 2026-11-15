import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-specific diagonal variances model intents whose encoder spread differs substantially across classes. Shrinking each class variance toward the global diagonal covariance avoids unstable estimates for small classes while retaining heteroscedastic distance information."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    z = np.asarray(ev.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    classes, yi = np.unique(y, return_inverse=True)
    c = classes.size
    n, d = x.shape
    counts = np.bincount(yi, minlength=c).astype(np.float64)

    means = np.zeros((c, d), dtype=np.float64)
    np.add.at(means, yi, x)
    means /= np.maximum(counts[:, None], 1.0)

    centered = x - means[yi]
    within_ss = np.zeros((c, d), dtype=np.float64)
    np.add.at(within_ss, yi, centered * centered)
    class_var = within_ss / np.maximum(counts[:, None] - 1.0, 1.0)

    global_var = np.var(x, axis=0)
    scale = max(float(np.median(global_var)), 1e-8)
    global_var = np.maximum(global_var, scale * 1e-4)

    shrink = 12.0
    var = (within_ss + shrink * global_var[None, :]) / (
        np.maximum(counts[:, None] - 1.0, 0.0) + shrink
    )
    var = np.maximum(var, scale * 1e-5)
    inv_var = 1.0 / var

    linear = means * inv_var
    offset = np.sum(means * linear, axis=1) + np.sum(np.log(var), axis=1)

    scores = np.empty(z.shape[0], dtype=np.float64)
    block = 512
    for start in range(0, z.shape[0], block):
        q = z[start:start + block]
        quad = (q * q) @ inv_var.T - 2.0 * (q @ linear.T) + offset[None, :]
        scores[start:start + q.shape[0]] = 0.5 * np.min(quad, axis=1)

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
