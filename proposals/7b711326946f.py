import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "A predicted intent can have an ordinary total diagonal Mahalanobis radius while "
    "containing one or a few extreme violations of coordinates that are stable for that intent. "
    "A train-calibrated fourth-moment residual score emphasizes these sparse class-specific deviations."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013", "m_0206"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.feats, dtype=np.float64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    global_mean = np.mean(x, axis=0)
    global_var = np.var(x, axis=0)
    positive_global = global_var[global_var > 0.0]
    base_var = float(np.median(positive_global)) if positive_global.size else 1.0
    global_var = np.maximum(global_var, base_var * 1e-6)

    counts = np.bincount(y, minlength=k).astype(np.float64)
    sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sums, y, x)
    means = np.empty((k, d), dtype=np.float64)
    for c in range(k):
        if counts[c] > 0:
            means[c] = sums[c] / counts[c]
        else:
            means[c] = global_mean

    sq_sums = np.zeros((k, d), dtype=np.float64)
    centered_train = x - means[y]
    np.add.at(sq_sums, y, centered_train * centered_train)

    variances = np.empty((k, d), dtype=np.float64)
    for c in range(k):
        if counts[c] > 1:
            class_var = sq_sums[c] / counts[c]
        else:
            class_var = global_var
        shrink = counts[c] / (counts[c] + 24.0)
        variances[c] = shrink * class_var + (1.0 - shrink) * global_var
    variances = np.maximum(variances, base_var * 1e-6)

    train_resid = (x - means[y]) / np.sqrt(variances[y])
    train_resid = np.clip(train_resid, -100.0, 100.0)
    train_raw = np.mean(train_resid ** 4, axis=1)

    global_center = float(np.median(train_raw))
    global_scale = float(
        np.quantile(train_raw, 0.90) - np.quantile(train_raw, 0.50)
    )
    global_scale = max(global_scale, 1e-6)

    centers = np.full(k, global_center, dtype=np.float64)
    scales = np.full(k, global_scale, dtype=np.float64)
    for c in range(k):
        values = train_raw[y == c]
        if values.size >= 4:
            centers[c] = np.median(values)
            scale = np.quantile(values, 0.90) - centers[c]
            scales[c] = max(float(scale), global_scale * 0.1, 1e-6)

    pred = np.argmax(ev.logits, axis=1)
    eval_resid = (z - means[pred]) / np.sqrt(variances[pred])
    eval_resid = np.clip(eval_resid, -100.0, 100.0)
    raw = np.mean(eval_resid ** 4, axis=1)

    scores = (raw - centers[pred]) / scales[pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
