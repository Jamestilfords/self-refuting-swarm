import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A predicted intent may be close under a shared covariance while violating dimensions "
    "that are unusually stable for that specific intent. Shrunk diagonal class covariances "
    "retain class-specific variance structure without unstable full covariance estimates."
)
SOURCE = "https://cdn.amazon.science/64/48/4d1e599c4472859e27a9f7647786/few-shot-out-of-domain-intent-detection-with-covariance-corrected-mahalanobis-distance.pdf"
PARENTS = []
MEMORY_USED = ["m_0002", "m_0007"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.feats, dtype=np.float64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(np.float64)
    global_mean = x.mean(axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0)
    floor = np.maximum(np.median(global_var) * 1e-5, 1e-8)
    global_var = np.maximum(global_var, floor)

    sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sums, y, x)
    means = sums / np.maximum(counts[:, None], 1.0)
    means[counts == 0] = global_mean

    sq_sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sq_sums, y, x * x)
    raw_var = sq_sums / np.maximum(counts[:, None], 1.0) - means * means
    raw_var = np.maximum(raw_var, 0.0)

    shrink = counts / (counts + 32.0)
    variances = shrink[:, None] * raw_var + (1.0 - shrink[:, None]) * global_var
    variances = np.maximum(variances, floor)

    pred = np.argmax(ev.logits, axis=1)
    delta = z - means[pred]
    dist = np.sum((delta * delta) / variances[pred], axis=1) / float(d)
    return np.nan_to_num(dist, nan=0.0, posinf=1e12, neginf=0.0)
