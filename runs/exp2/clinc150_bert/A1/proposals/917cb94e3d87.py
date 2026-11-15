import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Low-variance directions of the pooled within-class covariance expose subtle deviations that "
    "dominant representation directions can hide. Conditioning the residual on the classifier's "
    "predicted intent flags inputs that are atypical for the intent the model assigns."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0007", "m_0009"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.pooled, dtype=np.float64)

    n_classes = tr.logits.shape[1]
    d = x.shape[1]
    means = np.zeros((n_classes, d), dtype=np.float64)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)

    centered = x - means[y]
    cov = (centered.T @ centered) / max(len(x) - n_classes, 1)
    vals, vecs = np.linalg.eigh(cov)

    n_trailing = min(128, d)
    vals = vals[:n_trailing]
    vecs = vecs[:, :n_trailing]
    floor = max(float(np.max(vals)) * 1e-6, 1e-8)
    inv_vals = 1.0 / np.maximum(vals, floor)

    train_proj = centered @ vecs
    train_raw = np.sum(train_proj * train_proj * inv_vals[None, :], axis=1)
    train_raw = np.log1p(np.maximum(train_raw, 0.0))

    medians = np.zeros(n_classes, dtype=np.float64)
    scales = np.ones(n_classes, dtype=np.float64)
    global_med = float(np.median(train_raw))
    global_scale = float(np.median(np.abs(train_raw - global_med))) + 1e-6
    for c in range(n_classes):
        values = train_raw[y == c]
        if len(values) >= 3:
            medians[c] = np.median(values)
            scales[c] = np.median(np.abs(values - medians[c])) + 1e-6
        else:
            medians[c] = global_med
            scales[c] = global_scale

    pred = np.argmax(ev.logits, axis=1)
    eval_centered = z - means[pred]
    eval_proj = eval_centered @ vecs
    raw = np.sum(eval_proj * eval_proj * inv_vals[None, :], axis=1)
    raw = np.log1p(np.maximum(raw, 0.0))
    return (raw - medians[pred]) / scales[pred]
