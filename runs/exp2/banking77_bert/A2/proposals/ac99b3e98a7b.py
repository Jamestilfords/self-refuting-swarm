import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A shrinkage diagonal Gaussian preserves per-intent feature constraints that a tied covariance discards while remaining stable for intents with limited examples. Class-conditional calibration turns predicted-intent negative log likelihood into a comparable surprise score across naturally broad and narrow intents."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0065"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    n, d = x.shape
    k = tr.logits.shape[1]

    counts = np.bincount(y, minlength=k).astype(float)
    total_mean = np.mean(x, axis=0)
    total_var = np.var(x, axis=0)
    total_var = np.maximum(total_var, 1e-8)

    sums = np.zeros((k, d), dtype=float)
    sq_sums = np.zeros((k, d), dtype=float)
    np.add.at(sums, y, x)
    np.add.at(sq_sums, y, x * x)

    safe_counts = np.maximum(counts, 1.0)[:, None]
    means = sums / safe_counts
    means[counts == 0] = total_mean

    raw_var = sq_sums / safe_counts - means * means
    raw_var = np.maximum(raw_var, 0.0)

    # Empirical-Bayes shrinkage: low-count intents borrow global variances.
    shrink = (counts / (counts + 32.0))[:, None]
    variances = shrink * raw_var + (1.0 - shrink) * total_var
    variances = np.maximum(variances, total_var * 1e-5 + 1e-8)

    def class_nll(values, classes):
        mu = means[classes]
        var = variances[classes]
        delta = values - mu
        return 0.5 * np.sum(
            delta * delta / var + np.log(var),
            axis=1,
        )

    train_nll = class_nll(x, y)

    class_sum = np.bincount(y, weights=train_nll, minlength=k)
    class_mean = class_sum / np.maximum(counts, 1.0)

    centered = train_nll - class_mean[y]
    class_sq = np.bincount(y, weights=centered * centered, minlength=k)
    class_std = np.sqrt(class_sq / np.maximum(counts - 1.0, 1.0))

    pooled_std = np.std(train_nll)
    pooled_std = max(float(pooled_std), 1e-6)
    class_std = np.where(
        counts >= 4.0,
        np.maximum(class_std, 1e-6),
        pooled_std,
    )

    pred = np.argmax(ev.logits, axis=1)
    eval_nll = class_nll(z, pred)
    scores = (eval_nll - class_mean[pred]) / class_std[pred]

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
