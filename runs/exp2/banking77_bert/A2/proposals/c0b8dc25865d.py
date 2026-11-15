import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A shrunk diagonal covariance preserves intent-specific feature variance while borrowing "
    "strength from the pooled distribution for small classes. OOS queries forced into a known "
    "intent should have unusually high class-calibrated surprise under their predicted intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0077", "m_0089"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.pooled, dtype=np.float64)

    _, d = x.shape
    k = tr.logits.shape[1]
    counts = np.bincount(y, minlength=k).astype(np.float64)

    global_mean = x.mean(axis=0)
    global_var = x.var(axis=0) + 1e-8

    means = np.zeros((k, d), dtype=np.float64)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)
    means[counts == 0] = global_mean

    centered = x - means[y]
    class_ss = np.zeros((k, d), dtype=np.float64)
    np.add.at(class_ss, y, centered * centered)
    class_var = class_ss / np.maximum(counts[:, None], 1.0)

    shrink = counts / (counts + 24.0)
    variances = shrink[:, None] * class_var + (1.0 - shrink[:, None]) * global_var
    variances = np.maximum(variances, 1e-8)
    inv_var = 1.0 / variances
    log_det = np.log(variances).sum(axis=1)

    def class_nll(a, cls):
        diff = a - means[cls]
        quad = np.sum(diff * diff * inv_var[cls], axis=1)
        return 0.5 * (quad + log_det[cls])

    train_nll = class_nll(x, y)
    loc = np.zeros(k, dtype=np.float64)
    scale = np.ones(k, dtype=np.float64)

    for c in range(k):
        vals = train_nll[y == c]
        if vals.size:
            loc[c] = np.median(vals)
            mad = np.median(np.abs(vals - loc[c]))
            scale[c] = max(1.4826 * mad, 1e-5)

    pred = np.argmax(ev.logits, axis=1)
    scores = (class_nll(z, pred) - loc[pred]) / scale[pred]
    return np.asarray(scores, dtype=np.float64)
