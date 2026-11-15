import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A predicted intent should assign high likelihood under its own feature-wise variance profile. "
    "Class-specific shrunk diagonal covariances preserve intent-dependent feature scales while avoiding unstable covariance estimates."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0033", "m_0045"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)
    pred = np.argmax(np.asarray(ev.logits), axis=1)

    n, d = x_train.shape
    k = tr.logits.shape[1]
    global_mean = np.mean(x_train, axis=0)
    global_var = np.var(x_train, axis=0) + 1e-6

    means = np.empty((k, d), dtype=float)
    variances = np.empty((k, d), dtype=float)
    train_center = np.empty(k, dtype=float)
    train_scale = np.empty(k, dtype=float)

    for c in range(k):
        xc = x_train[labels == c]
        nc = len(xc)
        if nc == 0:
            means[c] = global_mean
            variances[c] = global_var
            train_center[c] = 0.0
            train_scale[c] = 1.0
            continue

        mu = np.mean(xc, axis=0)
        if nc > 1:
            class_var = np.var(xc, axis=0)
        else:
            class_var = global_var

        shrink = nc / (nc + 20.0)
        var = shrink * class_var + (1.0 - shrink) * global_var + 1e-6
        means[c] = mu
        variances[c] = var

        diff = xc - mu
        nll = 0.5 * np.sum((diff * diff) / var + np.log(var), axis=1)
        train_center[c] = np.median(nll)
        train_scale[c] = np.median(np.abs(nll - train_center[c])) * 1.4826 + 1e-6

    scores = np.empty(len(x_eval), dtype=float)
    for c in range(k):
        idx = np.flatnonzero(pred == c)
        if len(idx) == 0:
            continue
        diff = x_eval[idx] - means[c]
        nll = 0.5 * np.sum(
            (diff * diff) / variances[c] + np.log(variances[c]), axis=1
        )
        scores[idx] = (nll - train_center[c]) / train_scale[c]

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
