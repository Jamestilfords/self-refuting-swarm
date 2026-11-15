import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query that the classifier assigns to an intent should also resemble that intent under its own feature-wise variance profile. Predicted-class diagonal likelihood penalizes samples that are implausible for the selected intent, while shrinkage stabilizes variance estimates for small classes."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    n, d = x.shape
    k = tr.logits.shape[1]
    global_mean = np.mean(x, axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0)
    floor = max(float(np.median(global_var)) * 1e-6, 1e-10)
    global_var = np.maximum(global_var, floor)

    means = np.tile(global_mean, (k, 1))
    variances = np.tile(global_var, (k, 1))
    counts = np.zeros(k, dtype=np.float64)

    for c in range(k):
        xc = x[y == c]
        count = xc.shape[0]
        if count == 0:
            continue
        counts[c] = count
        mean = np.mean(xc, axis=0)
        if count > 1:
            class_var = np.mean((xc - mean) ** 2, axis=0)
        else:
            class_var = global_var

        shrink = float(count) / (float(count) + 25.0)
        means[c] = mean
        variances[c] = np.maximum(
            shrink * class_var + (1.0 - shrink) * global_var,
            floor,
        )

    pred = np.argmax(np.asarray(ev.logits), axis=1)
    mean = means[pred]
    var = variances[pred]

    residual = z - mean
    mahal = np.sum((residual * residual) / var, axis=1)
    logdet = np.sum(np.log(var), axis=1)

    prior = (counts[pred] + 1.0) / (float(n) + float(k))
    return 0.5 * (mahal + logdet) - np.log(prior)
