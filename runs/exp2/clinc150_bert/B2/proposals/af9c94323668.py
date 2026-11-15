import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A shrunk diagonal covariance per intent captures feature dimensions whose in-scope variation is "
    "class-specific, while class-conditional empirical tail calibration makes distances comparable across intents."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0099", "m_0117"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    z = np.asarray(ev.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(np.float64)
    global_mean = np.mean(x, axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0)
    global_var = np.maximum(global_var, 1e-8)

    sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sums, y, x)
    means = np.repeat(global_mean[None, :], k, axis=0)
    present = counts > 0
    means[present] = sums[present] / counts[present, None]

    sq_sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sq_sums, y, x * x)
    class_var = np.repeat(global_var[None, :], k, axis=0)
    class_var[present] = (
        sq_sums[present] / counts[present, None] - means[present] ** 2
    )
    class_var = np.maximum(class_var, 0.0)

    shrink = counts / (counts + 20.0)
    variances = shrink[:, None] * class_var + (1.0 - shrink[:, None]) * global_var
    variances = np.maximum(variances, 1e-8)

    train_dist = np.sum((x - means[y]) ** 2 / variances[y], axis=1)
    sorted_dist = []
    for c in range(k):
        if counts[c] > 0:
            sorted_dist.append(np.sort(train_dist[y == c]))
        else:
            sorted_dist.append(np.sort(train_dist))

    pred = np.argmax(ev.logits, axis=1)
    out = np.empty(z.shape[0], dtype=np.float64)
    for c in range(k):
        idx = np.flatnonzero(pred == c)
        if idx.size == 0:
            continue
        dist = np.sum((z[idx] - means[c]) ** 2 / variances[c], axis=1)
        ref = sorted_dist[c]
        tail = ref.size - np.searchsorted(ref, dist, side="left")
        out[idx] = -np.log((tail + 1.0) / (ref.size + 1.0))

    return out
