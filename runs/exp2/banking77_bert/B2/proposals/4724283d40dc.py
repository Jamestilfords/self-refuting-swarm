import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Low-variance directions of the pooled within-intent covariance encode constraints that "
    "known intents satisfy. A predicted-intent-conditioned, train-standardized trailing-space "
    "residual flags queries that violate those constraints without requiring proximity to another class."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0061"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(np.float64)
    sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sums, y, x)
    means = sums / np.maximum(counts[:, None], 1.0)

    centered = x - means[y]
    cov = (centered.T @ centered) / max(x.shape[0] - k, 1)
    cov.flat[:: d + 1] += 1e-6 * (np.trace(cov) / max(d, 1) + 1.0)

    vals, vecs = np.linalg.eigh(cov)
    q = min(128, max(16, d // 6))
    low = vecs[:, :q]

    train_residual = np.sum((centered @ low) ** 2, axis=1)
    class_sum = np.bincount(y, weights=train_residual, minlength=k)
    class_mean = class_sum / np.maximum(counts, 1.0)

    deviations = train_residual - class_mean[y]
    class_var = np.bincount(y, weights=deviations ** 2, minlength=k)
    class_std = np.sqrt(class_var / np.maximum(counts - 1.0, 1.0))
    global_std = np.std(train_residual) + 1e-8
    class_std = np.where(class_std > 1e-8, class_std, global_std)

    pred = np.argmax(ev.logits, axis=1)
    ev_centered = z - means[pred]
    ev_residual = np.sum((ev_centered @ low) ** 2, axis=1)
    return (ev_residual - class_mean[pred]) / class_std[pred]
