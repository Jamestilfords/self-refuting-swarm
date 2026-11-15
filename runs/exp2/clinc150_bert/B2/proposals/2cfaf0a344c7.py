import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Class-mean directions capture intent-discriminative variation, so the complementary residual should contain only within-intent detail. "
    "A sparse tail score over standardized residual coordinates detects OOS examples that violate a few unsupported semantic directions rather than merely having large total residual norm."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0101", "m_0103", "m_0117"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    n, d = x.shape
    k = tr.logits.shape[1]
    global_mean = np.mean(x, axis=0)
    counts = np.bincount(y, minlength=k).astype(np.float64)

    means = np.empty((k, d), dtype=np.float64)
    for c in range(k):
        mask = y == c
        means[c] = np.mean(x[mask], axis=0) if np.any(mask) else global_mean

    between = means - global_mean
    _, singular_values, vt = np.linalg.svd(between, full_matrices=False)
    if singular_values.size:
        rank = int(np.sum(singular_values > singular_values[0] * 1e-10))
    else:
        rank = 0
    rank = min(rank, d - 1)

    basis = vt[:rank].T if rank > 0 else None

    def residual(a, classes):
        centered = a - means[classes]
        if basis is None:
            return centered
        return centered - (centered @ basis) @ basis.T

    train_residual = residual(x, y)
    global_var = np.mean(train_residual * train_residual, axis=0)
    global_var = np.maximum(global_var, 1e-8)

    class_var = np.empty((k, d), dtype=np.float64)
    prior_strength = 24.0
    for c in range(k):
        mask = y == c
        if np.any(mask):
            sum_sq = np.sum(train_residual[mask] * train_residual[mask], axis=0)
            class_var[c] = (sum_sq + prior_strength * global_var) / (
                counts[c] + prior_strength
            )
        else:
            class_var[c] = global_var
    class_var = np.maximum(class_var, 1e-8)

    tail_width = min(12, d)

    def sparse_tail(residuals, classes):
        standardized_sq = (residuals * residuals) / class_var[classes]
        largest = np.partition(standardized_sq, d - tail_width, axis=1)[:, -tail_width:]
        return np.mean(largest, axis=1)

    train_raw = sparse_tail(train_residual, y)
    centers = np.empty(k, dtype=np.float64)
    scales = np.empty(k, dtype=np.float64)
    for c in range(k):
        values = train_raw[y == c]
        if values.size == 0:
            values = train_raw
        center = np.median(values)
        mad = np.median(np.abs(values - center))
        centers[c] = center
        scales[c] = max(1.4826 * mad, 1e-6)

    predicted = np.argmax(ev.logits, axis=1)
    eval_raw = sparse_tail(residual(z, predicted), predicted)
    return (eval_raw - centers[predicted]) / scales[predicted]
