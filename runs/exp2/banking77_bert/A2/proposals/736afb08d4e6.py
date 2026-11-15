import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Predicted-intent diagonal Mahalanobis distance preserves class-specific feature variances that a tied covariance suppresses. Shrinkage toward global variance stabilizes intents with few examples while emphasizing dimensions that are unusually constrained within each intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0031", "m_0043"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    z = np.asarray(ev.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    k = tr.logits.shape[1]
    d = x.shape[1]
    global_mean = x.mean(axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0)
    floor = np.maximum(global_var * 1e-6, 1e-10)

    means = np.empty((k, d), dtype=np.float64)
    variances = np.empty((k, d), dtype=np.float64)
    shrinkage = 20.0

    for c in range(k):
        xc = x[y == c]
        n = xc.shape[0]
        if n == 0:
            means[c] = global_mean
            variances[c] = global_var + floor
        else:
            mu = xc.mean(axis=0)
            within = np.mean((xc - mu) ** 2, axis=0)
            means[c] = mu
            variances[c] = (
                n * within + shrinkage * global_var
            ) / (n + shrinkage) + floor

    pred = np.argmax(ev.logits, axis=1)
    delta = z - means[pred]
    dist = np.mean((delta * delta) / variances[pred], axis=1)
    return np.nan_to_num(dist, nan=0.0, posinf=1e12, neginf=0.0)
