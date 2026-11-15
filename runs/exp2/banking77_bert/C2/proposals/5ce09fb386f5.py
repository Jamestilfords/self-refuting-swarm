import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query assigned to an intent should lie near that intent's own feature distribution, whose variances can differ substantially across intents. Shrinking each class-diagonal covariance toward the global diagonal covariance stabilizes estimates for rare classes while preserving class-specific geometry."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.pooled, dtype=float)
    k = tr.logits.shape[1]
    d = x.shape[1]

    global_mean = np.mean(x, axis=0)
    global_var = np.var(x, axis=0) + 1e-6

    means = np.empty((k, d), dtype=float)
    variances = np.empty((k, d), dtype=float)
    centers = np.empty(k, dtype=float)
    scales = np.empty(k, dtype=float)

    for c in range(k):
        xc = x[y == c]
        n_c = xc.shape[0]
        if n_c == 0:
            means[c] = global_mean
            variances[c] = global_var
            centers[c] = 0.0
            scales[c] = 1.0
            continue

        mu = np.mean(xc, axis=0)
        raw_var = np.var(xc, axis=0)
        shrink = n_c / (n_c + 32.0)
        var = shrink * raw_var + (1.0 - shrink) * global_var + 1e-6

        train_dist = np.sum((xc - mu) ** 2 / var, axis=1)
        med = np.median(train_dist)
        q75 = np.percentile(train_dist, 75.0)
        scale = max(q75 - med, 1e-6)

        means[c] = mu
        variances[c] = var
        centers[c] = med
        scales[c] = scale

    predicted = np.argmax(ev.logits, axis=1)
    predicted = np.clip(predicted, 0, k - 1)
    out = np.empty(z.shape[0], dtype=float)

    for c in range(k):
        mask = predicted == c
        if np.any(mask):
            dist = np.sum((z[mask] - means[c]) ** 2 / variances[c], axis=1)
            out[mask] = (dist - centers[c]) / scales[c]

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
