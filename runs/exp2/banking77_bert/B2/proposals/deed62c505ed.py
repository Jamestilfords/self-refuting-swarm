import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Intent classes can occupy feature regions with substantially different per-dimension spreads, "
    "so a shrinkage diagonal class-conditional likelihood can reject queries that tied-covariance models overaccept."
)
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
    counts = np.bincount(y, minlength=k).astype(np.float64)

    global_mean = x.mean(axis=0)
    global_var = x.var(axis=0)
    floor = max(float(np.mean(global_var)) * 1e-6, 1e-8)
    global_var = np.maximum(global_var, floor)

    sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sums, y, x)
    means = np.repeat(global_mean[None, :], k, axis=0)
    present = counts > 0
    means[present] = sums[present] / counts[present, None]

    variances = np.empty((k, d), dtype=np.float64)
    for c in range(k):
        if counts[c] <= 1:
            variances[c] = global_var
        else:
            xc = x[y == c] - means[c]
            class_var = np.mean(xc * xc, axis=0)
            shrink = counts[c] / (counts[c] + 32.0)
            variances[c] = shrink * class_var + (1.0 - shrink) * global_var

    variances = np.maximum(variances, floor)
    inv_var = 1.0 / variances

    # Quadratic diagonal Gaussian terms computed by matrix products, avoiding
    # an eval-items by classes by dimensions intermediate.
    linear = means * inv_var
    constant = np.sum(means * linear, axis=1) + np.sum(np.log(variances), axis=1)
    quadratic = (z * z) @ inv_var.T - 2.0 * (z @ linear.T) + constant[None, :]

    # Equal class weighting avoids letting frequent intents dominate support.
    return 0.5 * np.min(quadratic, axis=1) / float(d)
