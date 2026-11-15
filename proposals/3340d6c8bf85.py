import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A predicted intent should satisfy its own coordinate-wise variance constraints, but OOS inputs often "
    "produce many moderate residual violations rather than only a large total Mahalanobis radius."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013", "m_0094", "m_0027"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    z = np.asarray(ev.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(np.float64)
    sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sums, y, x)
    means = sums / np.maximum(counts[:, None], 1.0)

    global_mean = x.mean(axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0)
    global_var = np.maximum(global_var, 1e-6)

    centered = x - means[y]
    sq_sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sq_sums, y, centered * centered)
    class_var = sq_sums / np.maximum(counts[:, None], 1.0)

    shrink = counts / (counts + 24.0)
    variances = shrink[:, None] * class_var + (1.0 - shrink[:, None]) * global_var
    variances = np.maximum(variances, 1e-6)

    train_std = np.abs(centered) / np.sqrt(variances[y])
    train_excess = np.logaddexp(0.0, train_std - 1.5).mean(axis=1)

    med = np.zeros(k, dtype=np.float64)
    scale = np.zeros(k, dtype=np.float64)
    for c in range(k):
        vals = train_excess[y == c]
        if vals.size:
            q25, q50, q75 = np.percentile(vals, [25.0, 50.0, 75.0])
            med[c] = q50
            scale[c] = max(q75 - q25, 1e-4)
        else:
            med[c] = np.median(train_excess)
            scale[c] = max(
                np.percentile(train_excess, 75.0) - np.percentile(train_excess, 25.0),
                1e-4,
            )

    pred = np.argmax(ev.logits, axis=1)
    eval_std = np.abs(z - means[pred]) / np.sqrt(variances[pred])
    eval_excess = np.logaddexp(0.0, eval_std - 1.5).mean(axis=1)

    return (eval_excess - med[pred]) / scale[pred]
