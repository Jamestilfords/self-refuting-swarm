import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "An in-scope query's classifier-predicted intent should usually agree with its nearest "
    "class prototype in representation space. OOS queries can be forced into a logit class "
    "while lying substantially closer to another intent prototype."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0031"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    n_classes = tr.logits.shape[1]
    dim = x.shape[1]

    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    sums = np.zeros((n_classes, dim), dtype=np.float64)
    np.add.at(sums, y, x)

    global_mean = x.mean(axis=0)
    means = np.empty((n_classes, dim), dtype=np.float64)
    present = counts > 0
    means[present] = sums[present] / counts[present, None]
    means[~present] = global_mean

    residuals = x - means[y]
    within_var = np.mean(residuals * residuals, axis=0)
    global_var = np.var(x, axis=0)
    var = 0.9 * within_var + 0.1 * global_var
    var = np.maximum(var, 1e-8 * max(float(np.mean(global_var)), 1.0))
    inv_std = 1.0 / np.sqrt(var)

    zw = z * inv_std
    mw = means * inv_std
    z_norm = np.sum(zw * zw, axis=1, keepdims=True)
    m_norm = np.sum(mw * mw, axis=1)
    distances = z_norm + m_norm[None, :] - 2.0 * (zw @ mw.T)
    distances = np.maximum(distances, 0.0)

    predicted = np.argmax(ev.logits, axis=1)
    predicted_distance = distances[np.arange(z.shape[0]), predicted]
    nearest_distance = np.min(distances, axis=1)

    return np.maximum(predicted_distance - nearest_distance, 0.0)
