import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A class-specific shrunk diagonal covariance captures intent dimensions that are unusually stable, "
    "while the mean of the largest standardized residuals emphasizes sparse violations hidden by total distance."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    d = x_train.shape[1]
    eps = 1e-8
    tail_dims = min(24, d)

    global_mean = np.mean(x_train, axis=0)
    global_var = np.var(x_train, axis=0) + eps

    means = np.tile(global_mean, (k, 1))
    variances = np.tile(global_var, (k, 1))
    counts = np.bincount(y, minlength=k).astype(np.float64)

    for c in range(k):
        mask = y == c
        n_c = int(np.sum(mask))
        if n_c == 0:
            continue
        xc = x_train[mask]
        mean_c = np.mean(xc, axis=0)
        var_c = np.mean((xc - mean_c) ** 2, axis=0)
        shrink = n_c / (n_c + 32.0)
        means[c] = mean_c
        variances[c] = shrink * var_c + (1.0 - shrink) * global_var + eps

    def tail_residual(x, classes):
        out = np.empty(x.shape[0], dtype=np.float64)
        for c in range(k):
            mask = classes == c
            if not np.any(mask):
                continue
            z2 = (x[mask] - means[c]) ** 2 / variances[c]
            if tail_dims < d:
                largest = np.partition(z2, d - tail_dims, axis=1)[:, -tail_dims:]
                out[mask] = np.mean(largest, axis=1)
            else:
                out[mask] = np.mean(z2, axis=1)
        return out

    train_tail = tail_residual(x_train, y)
    class_center = np.zeros(k, dtype=np.float64)
    class_scale = np.ones(k, dtype=np.float64)
    for c in range(k):
        vals = train_tail[y == c]
        if vals.size == 0:
            vals = train_tail
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        class_center[c] = med
        class_scale[c] = max(1.4826 * mad, 0.05)

    predicted = np.argmax(ev.logits, axis=1)
    raw = tail_residual(x_eval, predicted)
    scores = (raw - class_center[predicted]) / class_scale[predicted]

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
