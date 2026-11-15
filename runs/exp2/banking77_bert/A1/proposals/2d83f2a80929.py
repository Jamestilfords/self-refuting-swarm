import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "OOS queries can violate a small number of tightly constrained, intent-specific representation coordinates even when their overall distance is moderate. A shrunk per-class diagonal variance model highlights these unusually large studentized residuals without collapsing them into a standard Gaussian density."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k_classes = tr.logits.shape[1]
    d = x.shape[1]

    global_mean = np.mean(x, axis=0)
    global_var = np.var(x, axis=0)
    positive = global_var[global_var > 0]
    base = float(np.median(positive)) if positive.size else 1.0
    floor = max(base * 1e-6, 1e-8)

    means = np.empty((k_classes, d), dtype=float)
    variances = np.empty((k_classes, d), dtype=float)

    for c in range(k_classes):
        xc = x[y == c]
        n_c = xc.shape[0]
        if n_c == 0:
            means[c] = global_mean
            variances[c] = global_var
            continue
        mean_c = np.mean(xc, axis=0)
        var_c = np.mean((xc - mean_c) ** 2, axis=0)
        shrink = 24.0 / (n_c + 24.0)
        means[c] = mean_c
        variances[c] = (1.0 - shrink) * var_c + shrink * global_var

    variances = np.maximum(variances, floor)
    predicted = np.argmax(ev.logits, axis=1)
    residual_sq = (z - means[predicted]) ** 2 / variances[predicted]

    top_count = min(8, d)
    largest = np.partition(residual_sq, d - top_count, axis=1)[:, -top_count:]
    return np.log1p(np.mean(largest, axis=1))
