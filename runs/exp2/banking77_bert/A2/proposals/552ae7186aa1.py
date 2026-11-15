import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Queries forced into a high-confidence known intent can still lie unusually far from that predicted "
    "intent's representation prototype. Class-conditional tail calibration accounts for natural differences "
    "in within-intent spread."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0053", "m_0057"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    classes = np.unique(y)
    k = int(np.max(classes)) + 1
    dim = x_train.shape[1]

    global_mean = x_train.mean(axis=0)
    centered = x_train - global_mean
    global_var = np.mean(centered * centered, axis=0) + 1e-6

    means = np.zeros((k, dim), dtype=float)
    variances = np.zeros((k, dim), dtype=float)
    counts = np.bincount(y, minlength=k).astype(float)

    for c in classes:
        xc = x_train[y == c]
        means[c] = xc.mean(axis=0)
        if len(xc) > 1:
            variances[c] = xc.var(axis=0)
        else:
            variances[c] = global_var

    shrink = counts[:, None] / (counts[:, None] + 20.0)
    variances = shrink * variances + (1.0 - shrink) * global_var[None, :]
    variances = np.maximum(variances, 1e-6)

    pred_train = np.argmax(tr.logits, axis=1)
    train_delta = x_train - means[pred_train]
    train_dist = np.mean((train_delta * train_delta) / variances[pred_train], axis=1)

    pred_eval = np.argmax(ev.logits, axis=1)
    eval_delta = x_eval - means[pred_eval]
    eval_dist = np.mean((eval_delta * eval_delta) / variances[pred_eval], axis=1)

    class_center = np.zeros(k, dtype=float)
    class_scale = np.ones(k, dtype=float)
    for c in classes:
        vals = train_dist[pred_train == c]
        if len(vals) >= 4:
            class_center[c] = np.median(vals)
            q25, q75 = np.percentile(vals, [25.0, 75.0])
            class_scale[c] = max(q75 - q25, 1e-4)
        else:
            class_center[c] = np.median(train_dist)
            class_scale[c] = max(
                np.percentile(train_dist, 75.0) - np.percentile(train_dist, 25.0),
                1e-4,
            )

    scores = (eval_dist - class_center[pred_eval]) / class_scale[pred_eval]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
