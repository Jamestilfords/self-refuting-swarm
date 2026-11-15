import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Within-class trailing covariance directions represent tightly constrained variation that is "
    "largely ignored by dominant semantic factors. The largest standardized deviation in this "
    "low-variance subspace detects concentrated off-manifold departures from known intents."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0053"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    k = tr.logits.shape[1]
    d = x.shape[1]
    means = np.zeros((k, d), dtype=np.float64)
    counts = np.bincount(y, minlength=k).astype(np.float64)

    for c in range(k):
        if counts[c] > 0:
            means[c] = x[y == c].mean(axis=0)
    global_mean = x.mean(axis=0)
    means[counts == 0] = global_mean

    residuals = x - means[y]
    cov = (residuals.T @ residuals) / max(len(x) - k, 1)
    scale = float(np.trace(cov) / max(d, 1))
    cov.flat[:: d + 1] += max(scale, 1e-8) * 1e-6

    vals, vecs = np.linalg.eigh(cov)
    n_tail = min(64, d)
    tail_vals = np.maximum(vals[:n_tail], max(scale, 1e-8) * 1e-6)
    tail_vecs = vecs[:, :n_tail]

    pred = np.argmax(ev.logits, axis=1)
    centered = z - means[pred]
    coords = (centered @ tail_vecs) / np.sqrt(tail_vals)[None, :]

    return np.max(np.abs(coords), axis=1)
