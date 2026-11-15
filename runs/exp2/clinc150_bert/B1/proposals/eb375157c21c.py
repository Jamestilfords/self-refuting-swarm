import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A shrunk diagonal covariance models intent-specific tolerance independently per representation "
    "coordinate, while the average of the largest standardized residuals emphasizes localized semantic "
    "violations that a summed Mahalanobis score can dilute."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0124", "m_0135"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]
    d = x.shape[1]

    global_mean = np.mean(x, axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0) + 1e-6
    counts = np.bincount(y, minlength=k).astype(float)

    means = np.empty((k, d), dtype=float)
    variances = np.empty((k, d), dtype=float)

    for c in range(k):
        xc = x[y == c]
        nc = xc.shape[0]
        if nc == 0:
            means[c] = global_mean
            variances[c] = global_var
            continue

        mu = np.mean(xc, axis=0)
        means[c] = mu
        if nc > 1:
            local_var = np.mean((xc - mu) ** 2, axis=0)
        else:
            local_var = global_var

        shrink = nc / (nc + 24.0)
        variances[c] = shrink * local_var + (1.0 - shrink) * global_var + 1e-6

    top_n = min(24, d)

    def tail_residual(features, classes):
        resid = features - means[classes]
        standardized_sq = (resid * resid) / variances[classes]
        largest = np.partition(standardized_sq, d - top_n, axis=1)[:, -top_n:]
        return np.mean(largest, axis=1)

    train_raw = tail_residual(x, y)
    cal_mean = np.empty(k, dtype=float)
    cal_std = np.empty(k, dtype=float)

    overall_mean = float(np.mean(train_raw))
    overall_std = float(np.std(train_raw) + 1e-6)
    for c in range(k):
        values = train_raw[y == c]
        if values.size >= 2:
            cal_mean[c] = np.mean(values)
            cal_std[c] = np.std(values) + 1e-6
        else:
            cal_mean[c] = overall_mean
            cal_std[c] = overall_std

    routed = np.argmax(np.asarray(ev.logits, dtype=float), axis=1)
    raw = tail_residual(z, routed)
    return (raw - cal_mean[routed]) / cal_std[routed]
