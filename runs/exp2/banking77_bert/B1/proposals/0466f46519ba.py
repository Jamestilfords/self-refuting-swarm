import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "A known intent should satisfy its own coordinate-wise semantic tolerances, while OOS queries often violate one or a few tightly constrained representation dimensions. A shrunk per-class diagonal variance model makes these extreme residuals stable even for small intents."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0101", "m_0112", "m_0114"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.feats, dtype=float)

    k = tr.logits.shape[1]
    d = x.shape[1]
    counts = np.bincount(y, minlength=k).astype(float)
    sums = np.zeros((k, d), dtype=float)
    sums2 = np.zeros((k, d), dtype=float)
    np.add.at(sums, y, x)
    np.add.at(sums2, y, x * x)

    safe_counts = np.maximum(counts, 1.0)[:, None]
    means = sums / safe_counts
    raw_var = sums2 / safe_counts - means * means
    raw_var = np.maximum(raw_var, 0.0)

    global_mean = np.mean(x, axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0)
    floor = max(float(np.mean(global_var)) * 1e-6, 1e-8)
    global_var = np.maximum(global_var, floor)

    # Moderate empirical-Bayes shrinkage prevents unstable small-class variances.
    strength = 24.0
    variances = (
        counts[:, None] * raw_var + strength * global_var[None, :]
    ) / (counts[:, None] + strength)
    variances = np.maximum(variances, floor)

    def tail_residual(a, classes):
        mu = means[classes]
        va = variances[classes]
        q = (a - mu) ** 2 / va
        # Smooth maximum: emphasizes a few implausible coordinates rather than
        # treating many harmless moderate residuals as equally anomalous.
        temperature = 3.0
        qmax = np.max(q, axis=1, keepdims=True)
        return (
            qmax[:, 0]
            + np.log(np.mean(np.exp(np.minimum(temperature * (q - qmax), 0.0)), axis=1))
            / temperature
        )

    train_tail = tail_residual(x, y)
    centers = np.zeros(k, dtype=float)
    scales = np.ones(k, dtype=float)
    for c in range(k):
        vals = train_tail[y == c]
        if vals.size:
            med = np.median(vals)
            mad = np.median(np.abs(vals - med))
            centers[c] = med
            scales[c] = max(1.4826 * mad, 1e-5)

    predicted = np.argmax(ev.logits, axis=1)
    eval_tail = tail_residual(z, predicted)
    return (eval_tail - centers[predicted]) / scales[predicted]
