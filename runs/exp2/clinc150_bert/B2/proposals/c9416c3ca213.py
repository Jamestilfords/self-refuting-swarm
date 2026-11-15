import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Intent-specific diagonal variances preserve dimensions whose acceptable variation differs across "
    "intents, while shrinkage toward global variance stabilizes small classes. Predicted-class tail scaling "
    "makes these heteroscedastic distances comparable without fitting an expensive full covariance per class."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0120"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    xtr = np.asarray(tr.feats, dtype=np.float64)
    xev = np.asarray(ev.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    d = xtr.shape[1]

    global_mean = np.mean(xtr, axis=0)
    global_var = np.var(xtr, axis=0) + 1e-8
    means = np.empty((k, d), dtype=np.float64)
    variances = np.empty((k, d), dtype=np.float64)
    scales = np.empty(k, dtype=np.float64)

    for c in range(k):
        xc = xtr[y == c]
        n = xc.shape[0]
        if n == 0:
            means[c] = global_mean
            variances[c] = global_var
            scales[c] = float(d)
            continue

        mu = np.mean(xc, axis=0)
        if n > 1:
            class_var = np.var(xc, axis=0)
        else:
            class_var = global_var

        # A moderate empirical-Bayes shrinkage prevents rare intents from
        # producing unstable near-zero per-coordinate variances.
        weight = n / (n + 24.0)
        var = weight * class_var + (1.0 - weight) * global_var
        var = np.maximum(var, global_var * 1e-5 + 1e-10)

        means[c] = mu
        variances[c] = var

        own_dist = np.sum((xc - mu) ** 2 / var, axis=1) / d
        scales[c] = np.quantile(own_dist, 0.90) + 1e-8

    pred = np.argmax(ev.logits, axis=1)
    out = np.empty(xev.shape[0], dtype=np.float64)
    for c in range(k):
        idx = pred == c
        if np.any(idx):
            delta = xev[idx] - means[c]
            dist = np.sum(delta * delta / variances[c], axis=1) / d
            out[idx] = np.log1p(dist / scales[c])

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=0.0)
