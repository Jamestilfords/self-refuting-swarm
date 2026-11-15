import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Class-mean directions explain feature variation used to distinguish known intents, so the complementary residual captures content not useful for any in-scope decision. A class-conditional standardized tail score on that residual emphasizes unsupported deviations while allowing different intents to have different residual variability."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0014", "m_0007"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    q = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    k = tr.logits.shape[1]
    d = x.shape[1]
    eps = 1e-10

    global_mean = x.mean(axis=0)
    counts = np.bincount(y, minlength=k).astype(np.float64)
    sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sums, y, x)
    class_means = sums / np.maximum(counts[:, None], 1.0)
    class_means[counts == 0] = global_mean

    between = class_means - global_mean
    valid = counts > 0
    between = between[valid]

    if between.shape[0] > 1:
        _, singular, vt = np.linalg.svd(between, full_matrices=False)
        threshold = np.finfo(np.float64).eps * max(between.shape) * singular[0]
        rank = int(np.sum(singular > threshold))
    else:
        vt = np.empty((0, d), dtype=np.float64)
        rank = 0

    xc = x - global_mean
    qc = q - global_mean
    if rank > 0:
        basis = vt[:rank].T
        xr = xc - (xc @ basis) @ basis.T
        qr = qc - (qc @ basis) @ basis.T
    else:
        xr = xc
        qr = qc

    residual_means = np.zeros((k, d), dtype=np.float64)
    np.add.at(residual_means, y, xr)
    residual_means /= np.maximum(counts[:, None], 1.0)
    residual_means[counts == 0] = 0.0

    sq_sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sq_sums, y, xr * xr)
    variances = sq_sums / np.maximum(counts[:, None], 1.0) - residual_means * residual_means

    global_var = np.mean((xr - residual_means[y]) ** 2, axis=0)
    shrink = counts[:, None] / (counts[:, None] + 20.0)
    variances = shrink * np.maximum(variances, 0.0) + (1.0 - shrink) * global_var[None, :]
    variances = np.maximum(variances, eps)

    def tail_stat(residuals, labels):
        z2 = (residuals - residual_means[labels]) ** 2 / variances[labels]
        tail = min(32, d)
        if tail == d:
            return z2.mean(axis=1)
        return np.partition(z2, d - tail, axis=1)[:, -tail:].mean(axis=1)

    train_raw = tail_stat(xr, y)
    train_sum = np.bincount(y, weights=train_raw, minlength=k)
    train_mean = train_sum / np.maximum(counts, 1.0)

    centered = train_raw - train_mean[y]
    train_scale_sum = np.bincount(y, weights=centered * centered, minlength=k)
    train_scale = np.sqrt(train_scale_sum / np.maximum(counts, 1.0))
    fallback_scale = np.std(train_raw) + eps
    train_scale = np.where(train_scale > eps, train_scale, fallback_scale)

    pred = np.argmax(ev.logits, axis=1).astype(np.int64)
    pred = np.clip(pred, 0, k - 1)
    eval_raw = tail_stat(qr, pred)
    out = (eval_raw - train_mean[pred]) / train_scale[pred]

    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)
