import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query confidently predicted as a known intent should lie near that predicted intent's representation prototype. Class-specific radial normalization makes distances comparable across intents with different semantic spread."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0093"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]

    global_mean = np.mean(x, axis=0)
    means = np.empty((k, x.shape[1]), dtype=float)
    scales = np.empty(k, dtype=float)

    global_d2 = np.sum((x - global_mean) ** 2, axis=1)
    global_scale = max(float(np.median(global_d2)), 1e-12)

    for c in range(k):
        mask = y == c
        if np.any(mask):
            xc = x[mask]
            mu = np.mean(xc, axis=0)
            d2 = np.sum((xc - mu) ** 2, axis=1)
            means[c] = mu
            scales[c] = max(float(np.median(d2)), 0.05 * global_scale, 1e-12)
        else:
            means[c] = global_mean
            scales[c] = global_scale

    pred = np.argmax(ev.logits, axis=1)
    residual = z - means[pred]
    d2 = np.sum(residual * residual, axis=1)
    return np.log1p(d2 / scales[pred])
