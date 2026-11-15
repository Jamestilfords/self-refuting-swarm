import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query should be close to the classifier-selected intent under that intent's own feature-wise variability. Shrinking each class diagonal covariance toward the global variance stabilizes estimates for smaller intent classes while preserving heteroscedastic class structure."
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

    global_mean = np.mean(x, axis=0)
    global_var = np.var(x, axis=0)
    var_floor = np.maximum(np.median(global_var) * 1e-6, 1e-10)
    global_var = np.maximum(global_var, var_floor)

    means = np.empty((k, d), dtype=float)
    variances = np.empty((k, d), dtype=float)
    shrink_strength = 24.0

    for c in range(k):
        xc = x[y == c]
        n = xc.shape[0]
        if n == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            mu = np.mean(xc, axis=0)
            if n > 1:
                local_var = np.var(xc, axis=0)
            else:
                local_var = global_var
            weight = n / (n + shrink_strength)
            means[c] = mu
            variances[c] = np.maximum(
                weight * local_var + (1.0 - weight) * global_var,
                var_floor,
            )

    predicted = np.argmax(ev.logits, axis=1)
    delta = z - means[predicted]
    score_values = np.sum((delta * delta) / variances[predicted], axis=1)

    train_predicted = np.argmax(tr.logits, axis=1)
    train_delta = x - means[train_predicted]
    train_dist = np.sum(
        (train_delta * train_delta) / variances[train_predicted],
        axis=1,
    )
    center = np.median(train_dist)
    scale = np.median(np.abs(train_dist - center)) * 1.4826
    scale = max(scale, 1e-8)

    return (score_values - center) / scale
