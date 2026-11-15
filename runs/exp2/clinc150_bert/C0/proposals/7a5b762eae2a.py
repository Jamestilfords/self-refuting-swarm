import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query should be plausible under the diagonal covariance of the intent predicted by the classifier, rather than merely close to any intent. Calibrating each predicted class by its training likelihood distribution accounts for intents with naturally different within-class variation."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    k = tr.logits.shape[1]
    d = x.shape[1]
    counts = np.bincount(y, minlength=k).astype(float)
    safe_counts = np.maximum(counts, 1.0)

    sums = np.zeros((k, d), dtype=float)
    sq_sums = np.zeros((k, d), dtype=float)
    np.add.at(sums, y, x)
    np.add.at(sq_sums, y, x * x)

    global_mean = x.mean(axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0)
    scale = max(float(np.mean(global_var)), 1e-8)
    floor = max(1e-8, 1e-6 * scale)
    global_var = np.maximum(global_var, floor)

    means = sums / safe_counts[:, None]
    means[counts == 0] = global_mean

    raw_var = sq_sums / safe_counts[:, None] - means * means
    raw_var = np.maximum(raw_var, 0.0)

    shrink = 25.0
    variances = (
        counts[:, None] * raw_var + shrink * global_var[None, :]
    ) / (counts[:, None] + shrink)
    variances = np.maximum(variances, floor)

    inv_var = 1.0 / variances
    log_det = np.sum(np.log(variances), axis=1)

    train_centered = x - means[y]
    train_nll = 0.5 * (
        np.sum(train_centered * train_centered * inv_var[y], axis=1)
        + log_det[y]
    )

    class_sum = np.bincount(y, weights=train_nll, minlength=k)
    class_mean = class_sum / safe_counts

    centered_train = train_nll - class_mean[y]
    class_sq = np.bincount(y, weights=centered_train * centered_train, minlength=k)
    class_std = np.sqrt(class_sq / safe_counts)
    fallback_std = max(float(np.std(train_nll)), 1e-6)
    class_std = np.maximum(class_std, 0.05 * fallback_std)

    pred = np.argmax(ev.logits, axis=1).astype(int)
    diff = z - means[pred]
    eval_nll = 0.5 * (
        np.sum(diff * diff * inv_var[pred], axis=1) + log_det[pred]
    )

    return (eval_nll - class_mean[pred]) / class_std[pred]
