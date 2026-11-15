import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "OOS queries often allocate an unusually large share of their within-class residual energy "
    "to tightly constrained trailing covariance directions. A train-calibrated trailing-to-bulk "
    "spectral residual ratio detects this directional mismatch rather than only total distance."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0017"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.feats, dtype=np.float64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(np.float64)
    means = np.zeros((k, d), dtype=np.float64)
    np.add.at(means, y, x)
    valid = counts > 0
    means[valid] /= counts[valid, None]
    global_mean = x.mean(axis=0)
    means[~valid] = global_mean

    centered = x - means[y]
    denom = max(x.shape[0] - int(valid.sum()), 1)
    cov = (centered.T @ centered) / float(denom)
    trace_scale = np.trace(cov) / max(d, 1)
    cov.flat[::d + 1] += max(trace_scale, 1e-8) * 1e-4

    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, max(trace_scale, 1e-8) * 1e-6)
    q = min(64, max(8, d // 4))
    inv_root = 1.0 / np.sqrt(vals)

    def spectral_ratio(a, classes):
        residual = a - means[classes]
        whitened = (residual @ vecs) * inv_root
        sq = whitened * whitened
        trailing = sq[:, :q].mean(axis=1)
        bulk = sq[:, q:].mean(axis=1) if q < d else sq.mean(axis=1)
        return np.log1p(trailing) - np.log1p(bulk)

    train_raw = spectral_ratio(x, y)
    loc = np.zeros(k, dtype=np.float64)
    scale = np.ones(k, dtype=np.float64)
    fallback_loc = np.median(train_raw)
    fallback_scale = np.median(np.abs(train_raw - fallback_loc)) * 1.4826 + 1e-6

    for c in range(k):
        values = train_raw[y == c]
        if values.size >= 4:
            med = np.median(values)
            mad = np.median(np.abs(values - med)) * 1.4826
            loc[c] = med
            scale[c] = max(mad, 1e-6)
        else:
            loc[c] = fallback_loc
            scale[c] = fallback_scale

    predicted = np.argmax(ev.logits, axis=1)
    raw = spectral_ratio(z, predicted)
    scores = (raw - loc[predicted]) / scale[predicted]
    scores = np.nan_to_num(scores, nan=0.0, posinf=50.0, neginf=-50.0)
    return np.clip(scores, -50.0, 50.0)
